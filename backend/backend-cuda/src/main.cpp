#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <filesystem>
#include <thread>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <condition_variable>

#include <CLI/CLI.hpp>
#include <toml++/toml.hpp>
#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXConnectionState.h>
#include <msgpack.hpp>

#include "utils.hpp" // For base64_decode and get_user_cache_dir
#include "config.hpp"
#include "model.hpp"

// protobuf includes
#include "handshake.pb.h"
#include "ping.pb.h"
#include "l4m.pb.h"
#include "l4m_vision.pb.h"


std::map<int, std::vector<uint8_t>> load_merge_rules(const std::filesystem::path& path) {
    std::map<int, std::vector<uint8_t>> merge_rules;
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Failed to load vocabulary file: " + path.string());

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        if (line.empty() || line.find_first_not_of(" \t\n\v\f\r") == std::string::npos) continue;

        std::string::size_type pos = line.find(' ');
        if (pos == std::string::npos) throw std::runtime_error("Error on line " + std::to_string(line_number) + ": expected 2 parts, but found 1.");
        
        std::string b64_token = line.substr(0, pos);
        std::string rank_str = line.substr(pos + 1);

        try {
            merge_rules[std::stoi(rank_str)] = utils::base64_decode(b64_token);
        } catch (const std::exception& e) {
            throw std::runtime_error("Error parsing line " + std::to_string(line_number) + ": " + e.what());
        }
    }
    return merge_rules;
}

ModelMetadata parse_model_metadata(const std::filesystem::path& metadata_path) {
    if (!std::filesystem::exists(metadata_path)) {
        throw std::runtime_error("Metadata file not found at: " + metadata_path.string());
    }
    
    const auto metadata_dir = metadata_path.parent_path();

    toml::table tbl;
    try {
        tbl = toml::parse_file(metadata_path.string());
    } catch (const toml::parse_error& err) {
        throw std::runtime_error("Failed to parse model metadata TOML '" + metadata_path.string() + "': " + std::string(err.what()));
    }

    ModelMetadata metadata;

    // Helper to get a required value from a TOML table.
    auto get_required = [&](const toml::table& t, const std::string& key, const std::string& section) {
        auto node = t[key];
        if (!node) {
            throw std::runtime_error("Missing required key '" + key + "' in section '[" + section + "]' of TOML file: " + metadata_path.string());
        }
        return node;
    };
    
    // --- Parse top-level keys ---
    metadata.name = get_required(tbl, "name", "top-level").value_or("");
    metadata.description = get_required(tbl, "description", "top-level").value_or("");

    if (auto params_node = get_required(tbl, "parameters", "top-level").as_array()) {
        for (const auto& elem : *params_node) {
            metadata.parameters.push_back(elem.value_or(""));
        }
    }

    // --- Parse Architecture ---
    auto arch_data = get_required(tbl, "architecture", "top-level").as_table();
    auto rope_data = get_required(*arch_data, "rope", "architecture").as_table();
    
    metadata.architecture = {
        .type = get_required(*arch_data, "type", "architecture").value_or(""),
        .num_layers = get_required(*arch_data, "num_layers", "architecture").value_or(0),
        .num_query_heads = get_required(*arch_data, "num_query_heads", "architecture").value_or(0),
        .num_key_value_heads = get_required(*arch_data, "num_key_value_heads", "architecture").value_or(0),
        .head_size = get_required(*arch_data, "head_size", "architecture").value_or(0),
        .hidden_size = get_required(*arch_data, "hidden_size", "architecture").value_or(0),
        .intermediate_size = get_required(*arch_data, "intermediate_size", "architecture").value_or(0),
        .vocab_size = get_required(*arch_data, "vocab_size", "architecture").value_or(0),
        .use_qkv_bias = false, // Hardcoded
        .rms_norm_eps = 1e-05f, // Hardcoded
        .rope_factor = static_cast<float>(get_required(*rope_data, "factor", "architecture.rope").value_or(0.0)),
        .rope_high_frequency_factor = static_cast<float>(get_required(*rope_data, "high_frequency_factor", "architecture.rope").value_or(0.0)),
        .rope_low_frequency_factor = static_cast<float>(get_required(*rope_data, "low_frequency_factor", "architecture.rope").value_or(0.0)),
        .rope_theta = static_cast<float>(get_required(*rope_data, "theta", "architecture.rope").value_or(0.0)),
    };

    // --- Parse Tokenizer ---
    auto tokenizer_data = get_required(tbl, "tokenizer", "top-level").as_table();
    std::string vocab_file = get_required(*tokenizer_data, "vocabulary_file", "tokenizer").value_or("");
    auto vocab_full_path = metadata_dir / metadata.name / vocab_file;
    
    metadata.tokenizer.type = get_required(*tokenizer_data, "type", "tokenizer").value_or("");
    metadata.tokenizer.split_regex = get_required(*tokenizer_data, "split_regex", "tokenizer").value_or("");
    metadata.tokenizer.merge_table = load_merge_rules(vocab_full_path);

    if (auto special_tokens_node = get_required(*tokenizer_data, "special_tokens", "tokenizer").as_table()) {
        for (const auto& [k, v] : *special_tokens_node) {
            if (auto id = v.value<int>()) {
                metadata.tokenizer.special_tokens[*id] = std::string(k);
            } else {
                std::cerr << "Failed to parse special token ID for key: " << k << std::endl;
            }
        }
    }

    // --- Parse Template ---
    auto template_data = get_required(tbl, "template", "top-level").as_table();
    metadata.template_type = get_required(*template_data, "type", "template").value_or("");
    metadata.template_content = get_required(*template_data, "content", "template").value_or("");

    return metadata;
}

std::filesystem::path get_cache_dir(const std::optional<std::string>& cli_path, const toml::table& config_table) {
    if (cli_path && !cli_path->empty()) return *cli_path;
    if (auto node = config_table["cache_dir"]; node.is_string()) return node.as_string()->get();
    if (const char* env_pie_home = std::getenv("PIE_HOME")) {
        if (std::strlen(env_pie_home) > 0) return env_pie_home;
    }
    return utils::get_user_cache_dir() / "pie";
}

/**
 * @brief Registers this service with the central controller using ixwebsocket.
 *
 * This corrected version uses the asynchronous, callback-based API provided by the library.
 * It uses a state machine within the callback and a condition_variable to wait
 * for the registration process to complete.
 */
void register_with_controller(const AppConfig& config, const std::string& service_endpoint) {
    std::cout << "[Registration Thread] Attempting registration with the controller..." << std::endl;

    // --- State machine and synchronization setup ---
    enum class RegistrationState {
        Connecting,
        AwaitingAuthResponse,
        AwaitingRegisterResponse,
        Succeeded,
        Failed
    };

    RegistrationState state = RegistrationState::Connecting;
    std::mutex mtx;
    std::condition_variable cv;
    bool process_finished = false;
    std::string error_reason;

    // 1. Setup the WebSocket client
    ix::WebSocket webSocket;
    const std::string url = "ws://" + config.controller_host + ":" + std::to_string(config.controller_port);
    webSocket.setUrl(url);

    // 2. Define the callback for handling all WebSocket events
    webSocket.setOnMessageCallback(
        [&](const ix::WebSocketMessagePtr& msg) {
            std::unique_lock<std::mutex> lock(mtx);

            switch (msg->type) {
                case ix::WebSocketMessageType::Open:
                    std::cout << "[Registration Thread] Connection established. Sending authentication request..." << std::endl;
                    {
                        // Now that we are open, send the authentication request
                        msgpack::sbuffer sbuf;
                        msgpack::packer<msgpack::sbuffer> packer(sbuf);
                        packer.pack_map(3);
                        packer.pack("type"); packer.pack("internal_authenticate");
                        packer.pack("corr_id"); packer.pack(0);
                        packer.pack("token"); packer.pack(config.internal_auth_token);
                        webSocket.sendBinary(std::string(sbuf.data(), sbuf.size()));
                        state = RegistrationState::AwaitingAuthResponse;
                    }
                    break;

                case ix::WebSocketMessageType::Message:
                    if (state == RegistrationState::AwaitingAuthResponse) {
                        std::cout << "[Registration Thread] Received authentication response." << std::endl;
                        try {
                            msgpack::object_handle oh = msgpack::unpack(msg->str.data(), msg->str.size());
                            std::map<std::string, msgpack::object> response;
                            oh.get().convert(response);

                            if (response["successful"].as<bool>()) {
                                std::cout << "[Registration Thread] Authentication successful. Sending registration request..." << std::endl;
                                // Auth succeeded, now send registration request
                                msgpack::sbuffer sbuf;
                                msgpack::packer<msgpack::sbuffer> packer(sbuf);
                                packer.pack_map(5);
                                packer.pack("type"); packer.pack("attach_remote_service");
                                packer.pack("corr_id"); packer.pack(0);
                                packer.pack("endpoint"); packer.pack(service_endpoint);
                                packer.pack("service_name"); packer.pack(config.model_name);
                                packer.pack("service_type"); packer.pack("l4m");
                                webSocket.sendBinary(std::string(sbuf.data(), sbuf.size()));
                                state = RegistrationState::AwaitingRegisterResponse;
                            } else {
                                error_reason = "Authentication failed by controller: " + (response.count("result") ? response["result"].as<std::string>() : "reason unknown");
                                state = RegistrationState::Failed;
                                process_finished = true;
                            }
                        } catch (const std::exception& e) {
                            error_reason = "Failed to parse auth response: " + std::string(e.what());
                            state = RegistrationState::Failed;
                            process_finished = true;
                        }
                    } else if (state == RegistrationState::AwaitingRegisterResponse) {
                        std::cout << "[Registration Thread] Received registration response." << std::endl;
                         try {
                            msgpack::object_handle oh = msgpack::unpack(msg->str.data(), msg->str.size());
                            std::map<std::string, msgpack::object> response;
                            oh.get().convert(response);

                            if (response["successful"].as<bool>()) {
                                std::cout << "[Registration Thread] Successfully registered with the controller." << std::endl;
                                state = RegistrationState::Succeeded;
                            } else {
                                error_reason = "Controller could not attach backend: " + (response.count("result") ? response["result"].as<std::string>() : "reason unknown");
                                state = RegistrationState::Failed;
                            }
                        } catch (const std::exception& e) {
                            error_reason = "Failed to parse registration response: " + std::string(e.what());
                            state = RegistrationState::Failed;
                        }
                        process_finished = true;
                    }
                    break;

                case ix::WebSocketMessageType::Error:
                    error_reason = "Connection error: " + msg->errorInfo.reason;
                    state = RegistrationState::Failed;
                    process_finished = true;
                    break;

                case ix::WebSocketMessageType::Close:
                    if (state != RegistrationState::Succeeded) {
                         error_reason = "Connection closed unexpectedly. Code: " + std::to_string(msg->closeInfo.code);
                         state = RegistrationState::Failed;
                    }
                    process_finished = true;
                    break;

                default:
                    break;
            }

            if (process_finished) {
                cv.notify_one();
            }
        });

    // 3. Start the WebSocket client's background thread
    webSocket.start();

    // 4. Wait until the registration process is finished (or fails)
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return process_finished; });
    }
    webSocket.stop();

    // 5. Check the final state and log any errors
    if (state == RegistrationState::Failed) {
        std::cerr << "[Registration Thread] Registration process failed: " << error_reason << std::endl;
    }
}


// Runs the main ZMQ server loop, listening for and handling requests.
void run_zmq_server(zmq::socket_t& router, const AppConfig& config, const ModelMetadata& metadata)
{
    std::cout << "[ZMQ Server Thread] ZMQ server thread started." << std::endl;
    std::vector<std::string> protocols = {"l4m", "l4m-vision", "ping"};
    std::unordered_map<std::string, bool> connected_clients;


    // Initializing the models
    Model model(config, metadata);
    
    // The run() method contains the main application loop.
    model.run();

    while (true) {
        try {
            std::vector<zmq::message_t> multipart_msg;
            auto result = zmq::recv_multipart(router, std::back_inserter(multipart_msg));

            if (!result) {
                std::cerr << "[ZMQ Server Thread] Failed to receive message." << std::endl;
                continue;
            }

            const std::string client_identity = multipart_msg[0].to_string();

            // Handle handshake for new clients
            if (connected_clients.find(client_identity) == connected_clients.end()) {
                std::cout << "[ZMQ Server Thread] New client connection, performing handshake." << std::endl;
                handshake::Request hs_req;
                if (!hs_req.ParseFromString(multipart_msg[1].to_string())) {
                    std::cerr << "[ZMQ Server Thread] Invalid handshake message." << std::endl;
                    continue;
                }
                
                handshake::Response hs_res;
                for(const auto& p : protocols) {
                    hs_res.add_protocols(p);
                }
                
                zmq::message_t response_payload(hs_res.ByteSizeLong());
                hs_res.SerializeToArray(response_payload.data(), response_payload.size());

                router.send(zmq::const_buffer(client_identity.data(), client_identity.size()), zmq::send_flags::sndmore);
                router.send(response_payload, zmq::send_flags::none);

                connected_clients[client_identity] = true;
                continue;
            }
            
            // Handle requests from existing clients
            if (multipart_msg.size() != 3) {
                std::cerr << "[ZMQ Server Thread] Invalid message format from known client." << std::endl;
                continue;
            }
            
            const int protocol_idx = *static_cast<const uint8_t*>(multipart_msg[1].data());
            if (protocol_idx < 0 || protocol_idx >= protocols.size()) {
                std::cerr << "[ZMQ Server Thread] Invalid protocol index: " << protocol_idx << std::endl;
                continue;
            }
            const std::string& protocol = protocols[protocol_idx];
            const std::string& payload_str = multipart_msg[2].to_string();

            if (protocol == "l4m") {
                l4m::Request request;
                request.ParseFromString(payload_str);
                
                bool needs_response = false;
                l4m::Response response;
                response.set_correlation_id(request.correlation_id());

                if (request.has_get_info()) {
                    l4m::Response response;
                    response.set_correlation_id(request.correlation_id());
                    auto* info = response.mutable_get_info();
                    info->set_model_name(metadata.name);
                    info->set_kv_page_size(config.kv_page_size);
                    info->set_num_available_kv_pages(config.max_num_kv_pages);
                    info->set_num_available_embeddings(config.max_num_embeds);
                    
                    auto* tokenizer_info = info->mutable_tokenizer();
                    auto* merge_table_proto = tokenizer_info->mutable_merge_table();
                    auto* special_tokens_proto = tokenizer_info->mutable_special_tokens();

                    tokenizer_info->set_split_regex(metadata.tokenizer.split_regex);

                    // The protobuf `bytes` type corresponds to `std::string` in C++.
                    for (const auto& [rank, token_bytes] : metadata.tokenizer.merge_table) {
                        (*merge_table_proto)[static_cast<uint32_t>(rank)] = 
                            std::string(token_bytes.begin(), token_bytes.end());
                    }

                    for (const auto& [id, name] : metadata.tokenizer.special_tokens) {
                        (*special_tokens_proto)[name] = static_cast<uint32_t>(id);
                    }

                    router.send(zmq::const_buffer(client_identity.data(), client_identity.size()), zmq::send_flags::sndmore);
                    router.send(multipart_msg[1], zmq::send_flags::sndmore);
                    router.send(zmq::buffer(response.SerializeAsString()), zmq::send_flags::none);
                }  else if (request.has_allocate()) {
                    //std::cout << "[ZMQ Server Thread] Handling BatchAllocate request." << std::endl;
                    std::vector<Model::AllocateCommand> commands;
                    commands.reserve(request.allocate().items_size());
                    for (const auto& item : request.allocate().items()) {
                        commands.push_back({
                            static_cast<Model::ObjectKind>(item.kind()),
                            item.object_id_offset(),
                            item.count()
                        });
                    }
                    model.handle_allocate(commands);

                } else if (request.has_deallocate()) {
                    //std::cout << "[ZMQ Server Thread] Handling BatchDeallocate request." << std::endl;
                    std::vector<Model::DeallocateCommand> commands;
                    commands.reserve(request.deallocate().items_size());
                    for (const auto& item : request.deallocate().items()) {
                        commands.push_back({
                            static_cast<Model::ObjectKind>(item.kind()),
                            item.object_id_offset(),
                            item.count()
                        });
                    }
                    model.handle_deallocate(commands);

                } else if (request.has_embed_text()) {
                    //std::cout << "[ZMQ Server Thread] Handling BatchEmbedText request." << std::endl;
                    std::vector<Model::EmbedTextCommand> commands;
                    commands.reserve(request.embed_text().items_size());
                    for (const auto& item : request.embed_text().items()) {
                        commands.push_back({
                            item.embedding_id(),
                            item.token_id(),
                            item.position_id()
                        });
                    }
                    model.handle_embed_text(commands);

                } else if (request.has_fill_block()) {
                    //std::cout << "[ZMQ Server Thread] Handling BatchFillBlock request." << std::endl;
                    std::vector<Model::FillBlockCommand> commands;
                    commands.reserve(request.fill_block().items_size());
                    for (const auto& item : request.fill_block().items()) {
                        commands.push_back({
                            item.last_block_len(),
                            {item.context_block_ids().begin(), item.context_block_ids().end()},
                            {item.input_embedding_ids().begin(), item.input_embedding_ids().end()},
                            {item.output_embedding_ids().begin(), item.output_embedding_ids().end()}
                        });
                    }
                    model.handle_fill_block(commands);

                    needs_response = true;
                    auto* proto_response = response.mutable_batch_sync();

                } else if (request.has_mask_block()) {
                    //std::cout << "[ZMQ Server Thread] Handling BatchMaskBlock request." << std::endl;
                    std::vector<Model::MaskBlockCommand> commands;
                    commands.reserve(request.mask_block().items_size());
                    for (const auto& item : request.mask_block().items()) {
                        commands.push_back({
                            item.block_id(),
                            {item.mask().begin(), item.mask().end()}
                        });
                    }
                    model.handle_mask_block(commands);

                } else if (request.has_copy_block()) {
                    //std::cout << "[ZMQ Server Thread] Handling BatchCopyBlock request." << std::endl;
                    std::vector<Model::CopyBlockCommand> commands;
                    commands.reserve(request.copy_block().items_size());
                    for (const auto& item : request.copy_block().items()) {
                        commands.push_back({
                            item.source_block_id(),
                            item.destination_block_id(),
                            item.source_start(),
                            item.destination_start(),
                            item.length()
                        });
                    }
                    model.handle_copy_block(commands);

                } else if (request.has_decode_token_distribution()) {
                    //std::cout << "[ZMQ Server Thread] Handling BatchDecodeTokenDistribution request." << std::endl;
                    std::vector<Model::DecodeTokenDistributionCommand> commands;
                    commands.reserve(request.decode_token_distribution().items_size());
                    for (const auto& item : request.decode_token_distribution().items()) {
                        commands.push_back({
                            item.embedding_id(),
                            item.distribution_id()
                        });
                    }
                    model.handle_decode_token_distribution(commands);

                } else if (request.has_sample_top_k_request()) {
                    //std::cout << "[ZMQ Server Thread] Handling BatchSampleTopKRequest request." << std::endl;
                    needs_response = true;
                    
                    std::vector<Model::SampleTopKCommand> commands;
                    commands.reserve(request.sample_top_k_request().items_size());
                    for (const auto& item : request.sample_top_k_request().items()) {
                        commands.push_back({item.distribution_id(), item.k()});
                    }

                    std::vector<Model::SampleTopKResult> results = model.handle_sample_top_k(commands);

                    auto* proto_response = response.mutable_sample_top_k();
                    for (const auto& result_item : results) {
                        auto* proto_item = proto_response->add_items();
                        proto_item->mutable_token_ids()->Add(result_item.token_ids.begin(), result_item.token_ids.end());
                        proto_item->mutable_probabilities()->Add(result_item.probabilities.begin(), result_item.probabilities.end());
                    }

                } else {
                    std::cerr << "[ZMQ Server Thread] Unknown or empty L4M command." << std::endl;
                }

                if (needs_response) {
                    router.send(zmq::const_buffer(client_identity.data(), client_identity.size()), zmq::send_flags::sndmore);
                    router.send(multipart_msg[1], zmq::send_flags::sndmore);
                    router.send(zmq::buffer(response.SerializeAsString()), zmq::send_flags::none);
                }
                // TODO: Implement other l4m commands.

            } else if (protocol == "ping") {
                ping::Ping ping_req;
                ping_req.ParseFromString(payload_str);

                ping::Pong pong_res;
                pong_res.set_correlation_id(ping_req.correlation_id());
                pong_res.set_message("Pong:" + ping_req.message());

                router.send(zmq::const_buffer(client_identity.data(), client_identity.size()), zmq::send_flags::sndmore);
                router.send(multipart_msg[1], zmq::send_flags::sndmore);
                router.send(zmq::buffer(pong_res.SerializeAsString()), zmq::send_flags::none);
            }
            
        } catch (const zmq::error_t& e) {
            if (e.num() == ETERM) {
                std::cout << "[ZMQ Server Thread] Context terminated, shutting down." << std::endl;
                break;
            }
            std::cerr << "[ZMQ Server Thread] ZMQ Error: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[ZMQ Server Thread] An unexpected error occurred: " << e.what() << std::endl;
        }
    }
}






int main(int argc, char* argv[]) {
    // --- 1. Argument Parsing Setup ---
    CLI::App app{"PIE C++ Backend Service"};
    app.allow_config_extras(true);

    AppConfig cli_config;
    std::optional<std::string> config_filepath_opt;
    std::optional<std::string> cache_dir_opt;
    
    app.add_option("--config", config_filepath_opt, "Path to a TOML configuration file.")->check(CLI::ExistingFile);
    app.add_option("--host", cli_config.host, "The hostname to bind to.");
    app.add_option("--port", cli_config.port, "The port number to listen on.");
    app.add_option("--controller_host", cli_config.controller_host, "The controller hostname.");
    app.add_option("--controller_port", cli_config.controller_port, "The controller port number.");
    app.add_option("--internal_auth_token", cli_config.internal_auth_token, "The authentication token for the controller.");
    app.add_option("--model", cli_config.model_name, "The model name (e.g., 'llama-3.2-1b-instruct').");
    app.add_option("--cache_dir", cache_dir_opt, "The directory for caching models.");
    app.add_option("--kv_page_size", cli_config.kv_page_size, "The KV page size.");
    app.add_option("--dist_size", cli_config.dist_size, "The distribution size.");
    app.add_option("--max_num_kv_pages", cli_config.max_num_kv_pages, "The maximum number of KV pages.");
    app.add_option("--max_num_embeds", cli_config.max_num_embeds, "The maximum number of embeddings.");
    app.add_option("--device", cli_config.device, "The device to use (e.g., 'cuda:0').");
    app.add_option("--dtype", cli_config.dtype, "The data type (e.g., 'bfloat16').");

    CLI11_PARSE(app, argc, argv);

    try {
        // --- 2. Configuration Loading and Merging ---
        toml::table config_from_file;
        if (config_filepath_opt) {
            try {
                config_from_file = toml::parse_file(*config_filepath_opt);
            } catch (const toml::parse_error& err) {
                std::cerr << "Error decoding TOML file '" << *config_filepath_opt << "': " << err.what() << std::endl;
                return 1;
            }
        }

        AppConfig final_config;
        // (Configuration merging logic remains the same)
        final_config.host = app.count("--host") > 0 ? cli_config.host : config_from_file["host"].value_or(final_config.host);
        final_config.port = app.count("--port") > 0 ? cli_config.port : config_from_file["port"].value_or(final_config.port);
        final_config.controller_host = app.count("--controller_host") > 0 ? cli_config.controller_host : config_from_file["controller_host"].value_or(final_config.controller_host);
        final_config.controller_port = app.count("--controller_port") > 0 ? cli_config.controller_port : config_from_file["controller_port"].value_or(final_config.controller_port);
        
        if (app.count("--model") > 0) final_config.model_name = cli_config.model_name;
        else if (auto node = config_from_file["model"]; node.is_string()) final_config.model_name = node.as_string()->get();

        if (app.count("--internal_auth_token") > 0) final_config.internal_auth_token = cli_config.internal_auth_token;
        else if (auto node = config_from_file["internal_auth_token"]; node.is_string()) final_config.internal_auth_token = node.as_string()->get();

        final_config.cache_dir = get_cache_dir(cache_dir_opt, config_from_file);
        // (Other config assignments remain the same)

        std::cout << final_config << std::endl;

        // Parse the model metadata
        // path = final_config.cache_dir / f{final_config.model_name}-{final_config.version}.toml
        std::filesystem::path metadata_path = final_config.cache_dir / "models" / (final_config.model_name + ".toml");
        if (!std::filesystem::exists(metadata_path)) {
            std::cerr << "Metadata file not found at: " << metadata_path.string() << std::endl;
            return 1;
        }
        ModelMetadata model_metadata = parse_model_metadata(metadata_path);
        std::cout << model_metadata << std::endl;


        // --- 4. Start Services ---
        zmq::context_t context(1);
        zmq::socket_t router(context, zmq::socket_type::router);
        
        std::string service_endpoint;
        std::string service_endpoint_public;

        if (final_config.host == "localhost" || final_config.host == "127.0.0.1") {
            int unique_id = std::rand() % 9000 + 1000;
            service_endpoint = "ipc:///tmp/pie-service-" + std::to_string(unique_id);
            service_endpoint_public = service_endpoint;

            std::cout << "Using IPC endpoint: " << service_endpoint << std::endl;
        } else {
            service_endpoint = "tcp://*:" + std::to_string(final_config.port);
            service_endpoint_public = "tcp://" + final_config.host + ":" + std::to_string(final_config.port);
            std::cout << "Using TCP endpoint: " << service_endpoint << std::endl;
        }
        router.bind(service_endpoint);
        std::cout << "Server listening on " << service_endpoint << std::endl;


        std::thread zmq_thread(run_zmq_server, std::ref(router), std::cref(final_config), std::cref(model_metadata));
        std::thread register_thread(register_with_controller, std::cref(final_config), service_endpoint_public);

        std::cout << "Service is running. Press Ctrl+C to shut down." << std::endl;
        zmq_thread.join();
        register_thread.join();

        router.close();
        context.close();

        
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Server shutdown initiated." << std::endl;
    return 0;
}