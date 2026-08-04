#pragma once

// Host-side transformed expert pack for SSD streaming (GPT-OSS native Marlin).
//
// Keyed by the compile cache key. Path: `{compile_cache_dir}/{key}.experts`.
// Pack build appends one expert at a time so peak VRAM stays O(one expert).

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "expert_stream_cache.hpp"
#include "loader/safetensors.hpp"
#include "loader/rust_loader_bridge.hpp"
#include "loader/weight_store_codec.hpp"

namespace pie_cuda_driver {

inline constexpr char kExpertPackMagic[8] = {'P', 'I', 'E', 'E', 'X', 'P', 'K', '1'};
inline constexpr std::uint32_t kExpertPackVersion = 1;

// On-disk header layout (before the per-expert body):
//   magic | version_u32 | key_len_u32 | key_bytes | L_u32 | E_u32 | S_u32
//   | section_bytes[S]_u64 | body_hash_u64
inline constexpr std::size_t kExpertPackMagicBytes = sizeof(kExpertPackMagic);
inline constexpr std::size_t kExpertPackVersionBytes = sizeof(std::uint32_t);
inline constexpr std::size_t kExpertPackKeyLenBytes = sizeof(std::uint32_t);
inline constexpr std::size_t kExpertPackDimsBytes =
    3 * sizeof(std::uint32_t);  // num_layers, num_experts, sections_per_expert
inline constexpr std::size_t kExpertPackSectionBytesEntry =
    sizeof(std::uint64_t);
inline constexpr std::size_t kExpertPackHashBytes = sizeof(std::uint64_t);

inline std::filesystem::path expert_pack_dir()
{
    return rust_loader_compile_cache_dir();
}

inline std::filesystem::path expert_pack_path(const std::string& cache_key)
{
    return expert_pack_dir() / (cache_key + ".experts");
}

// Unique per builder so concurrent cold builds of the same cache_key do not
// truncate each other's in-progress temps. Final publish still renames to
// expert_pack_path(cache_key).
inline std::filesystem::path expert_pack_tmp_path(const std::string& cache_key)
{
    static std::atomic<std::uint64_t> seq{0};
    const auto n = seq.fetch_add(1, std::memory_order_relaxed);
    return expert_pack_dir() /
           (cache_key + "." + std::to_string(static_cast<long long>(::getpid())) +
            "." + std::to_string(n) + ".experts.tmp");
}

inline std::uint64_t expert_pack_header_bytes(
    const std::string& cache_key,
    std::size_t sections)
{
    return kExpertPackMagicBytes + kExpertPackVersionBytes +
           kExpertPackKeyLenBytes + cache_key.size() + kExpertPackDimsBytes +
           sections * kExpertPackSectionBytesEntry + kExpertPackHashBytes;
}

inline void remap_streamed_table_to_expert_pack(
    StreamedExpertTable& table,
    const std::filesystem::path& pack_path,
    const std::string& cache_key)
{
    const std::uint64_t hdr = expert_pack_header_bytes(
        cache_key, table.section_bytes.size());
    table.shard_paths.assign(1, pack_path);
    for (auto& entry : table.extents) {
        for (auto& sec : entry.sections) {
            sec.shard = 0;
            sec.file_offset += hdr;
        }
    }
}

inline std::uint64_t expert_pack_body_bytes(const StreamedExpertTable& table)
{
    if (table.slot_bytes == 0) {
        throw std::runtime_error(
            "expert pack: slot_bytes must be > 0 (aligned slot stride from "
            "the stream plan); refusing to infer size from raw section "
            "payloads");
    }
    return table.slot_bytes * static_cast<std::uint64_t>(table.num_layers) *
           static_cast<std::uint64_t>(table.num_experts);
}

// Canonical body checksum (weight_codec::kChunkBytes folds) so warm-path
// verify matches finalize regardless of how append_bytes was chunked.
inline bool expert_pack_hash_body_file(
    const std::filesystem::path& path,
    std::uint64_t header_bytes,
    std::uint64_t body_bytes,
    std::uint64_t* out_hash)
{
    if (out_hash == nullptr) return false;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(static_cast<std::streamoff>(header_bytes));
    if (!in) return false;
    std::uint64_t sum = weight_codec::kBlobHashSeed;
    if (body_bytes == 0) {
        *out_hash = sum;
        return true;
    }
    std::vector<char> buf(static_cast<std::size_t>(
        std::min(weight_codec::kChunkBytes, body_bytes)));
    std::uint64_t left = body_bytes;
    while (left > 0) {
        const std::uint64_t n =
            std::min<std::uint64_t>(left, weight_codec::kChunkBytes);
        if (buf.size() < static_cast<std::size_t>(n)) {
            buf.resize(static_cast<std::size_t>(n));
        }
        in.read(buf.data(), static_cast<std::streamsize>(n));
        if (!in || static_cast<std::uint64_t>(in.gcount()) != n) {
            return false;
        }
        sum = weight_codec::blob_hash_update(
            sum, buf.data(), static_cast<std::size_t>(n));
        left -= n;
    }
    *out_hash = sum;
    return true;
}

class ExpertPackWriter {
public:
    ExpertPackWriter(
        const std::string& cache_key,
        const StreamedExpertTable& table)
        : cache_key_(cache_key),
          table_(table),
          path_(expert_pack_path(cache_key)),
          tmp_(expert_pack_tmp_path(cache_key))
    {
        std::error_code ec;
        std::filesystem::create_directories(expert_pack_dir(), ec);
        const std::uint64_t body = expert_pack_body_bytes(table);
        {
            const auto space = std::filesystem::space(expert_pack_dir(), ec);
            const std::uint64_t need = body + (256ull << 20);
            if (!ec && space.available < need) {
                throw std::runtime_error(
                    "expert pack: not enough free disk for pack in " +
                    expert_pack_dir().string());
            }
        }
        os_.open(tmp_, std::ios::binary | std::ios::trunc);
        if (!os_) {
            throw std::runtime_error(
                "expert pack: failed to open " + tmp_.string());
        }
        header_bytes_ = expert_pack_header_bytes(
            cache_key, table.section_bytes.size());
        std::vector<char> pad(header_bytes_, 0);
        os_.write(pad.data(), static_cast<std::streamsize>(pad.size()));
        body_cursor_ = 0;
    }

    void append_bytes(const void* data, std::uint64_t nbytes)
    {
        if (nbytes == 0) return;
        os_.write(static_cast<const char*>(data),
                  static_cast<std::streamsize>(nbytes));
        if (!os_) {
            throw std::runtime_error("expert pack: write failed");
        }
        body_cursor_ += nbytes;
    }

    void append_zeros(std::uint64_t nbytes)
    {
        if (nbytes == 0) return;
        std::vector<char> z(static_cast<std::size_t>(std::min<std::uint64_t>(nbytes, 1 << 20)), 0);
        std::uint64_t left = nbytes;
        while (left > 0) {
            const std::uint64_t n = std::min<std::uint64_t>(left, z.size());
            append_bytes(z.data(), n);
            left -= n;
        }
    }

    void finalize()
    {
        const std::uint64_t expected = expert_pack_body_bytes(table_);
        if (body_cursor_ != expected) {
            throw std::runtime_error(
                "expert pack: body size mismatch (got " +
                std::to_string(body_cursor_) + ", expected " +
                std::to_string(expected) + ")");
        }
        os_.flush();
        os_.close();
        std::uint64_t hash = 0;
        if (!expert_pack_hash_body_file(
                tmp_, header_bytes_, expected, &hash)) {
            throw std::runtime_error(
                "expert pack: failed to hash body in " + tmp_.string());
        }
        std::fstream out(
            tmp_, std::ios::binary | std::ios::in | std::ios::out);
        if (!out) {
            throw std::runtime_error(
                "expert pack: failed to reopen " + tmp_.string() +
                " for header write");
        }
        out.seekp(0);
        out.write(kExpertPackMagic,
                  static_cast<std::streamsize>(kExpertPackMagicBytes));
        write_u32(out, kExpertPackVersion);
        write_u32(out, static_cast<std::uint32_t>(cache_key_.size()));
        out.write(cache_key_.data(),
                  static_cast<std::streamsize>(cache_key_.size()));
        write_u32(out, static_cast<std::uint32_t>(table_.num_layers));
        write_u32(out, static_cast<std::uint32_t>(table_.num_experts));
        write_u32(out, static_cast<std::uint32_t>(table_.sections_per_expert));
        for (std::uint64_t b : table_.section_bytes) {
            write_u64(out, b);
        }
        write_u64(out, hash);
        out.flush();
        if (!out) {
            throw std::runtime_error("expert pack: header write failed");
        }
        out.close();
        std::filesystem::rename(tmp_, path_);
    }

    void abandon()
    {
        os_.close();
        std::error_code ec;
        std::filesystem::remove(tmp_, ec);
    }

    const std::filesystem::path& path() const noexcept { return path_; }

    static bool exists_and_matches(
        const std::string& cache_key,
        const StreamedExpertTable& table)
    {
        const auto path = expert_pack_path(cache_key);
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        char magic[kExpertPackMagicBytes];
        in.read(magic, static_cast<std::streamsize>(kExpertPackMagicBytes));
        if (!in ||
            std::memcmp(magic, kExpertPackMagic, kExpertPackMagicBytes) != 0) {
            return false;
        }
        std::uint32_t ver = 0, key_len = 0;
        in.read(reinterpret_cast<char*>(&ver),
                static_cast<std::streamsize>(kExpertPackVersionBytes));
        in.read(reinterpret_cast<char*>(&key_len),
                static_cast<std::streamsize>(kExpertPackKeyLenBytes));
        if (!in || ver != kExpertPackVersion) return false;
        std::string key(key_len, '\0');
        in.read(key.data(), static_cast<std::streamsize>(key_len));
        if (!in || key != cache_key) return false;
        std::uint32_t L = 0, E = 0, S = 0;
        in.read(reinterpret_cast<char*>(&L),
                static_cast<std::streamsize>(sizeof(std::uint32_t)));
        in.read(reinterpret_cast<char*>(&E),
                static_cast<std::streamsize>(sizeof(std::uint32_t)));
        in.read(reinterpret_cast<char*>(&S),
                static_cast<std::streamsize>(sizeof(std::uint32_t)));
        if (!in || static_cast<int>(L) != table.num_layers ||
            static_cast<int>(E) != table.num_experts ||
            static_cast<int>(S) != table.sections_per_expert) {
            return false;
        }
        for (std::uint64_t expect : table.section_bytes) {
            std::uint64_t b = 0;
            in.read(reinterpret_cast<char*>(&b),
                    static_cast<std::streamsize>(kExpertPackSectionBytesEntry));
            if (!in || b != expect) return false;
        }
        std::uint64_t stored_hash = 0;
        in.read(reinterpret_cast<char*>(&stored_hash),
                static_cast<std::streamsize>(kExpertPackHashBytes));
        if (!in) return false;
        in.close();

        const std::uint64_t hdr = expert_pack_header_bytes(
            cache_key, table.section_bytes.size());
        const std::uint64_t body = expert_pack_body_bytes(table);
        std::error_code ec;
        const auto actual = std::filesystem::file_size(path, ec);
        if (ec || actual != hdr + body) return false;

        std::uint64_t got_hash = 0;
        if (!expert_pack_hash_body_file(path, hdr, body, &got_hash)) {
            return false;
        }
        return got_hash == stored_hash;
    }

private:
    static void write_u32(std::ostream& os, std::uint32_t v)
    {
        os.write(reinterpret_cast<const char*>(&v),
                 static_cast<std::streamsize>(sizeof(std::uint32_t)));
    }
    static void write_u64(std::ostream& os, std::uint64_t v)
    {
        os.write(reinterpret_cast<const char*>(&v),
                 static_cast<std::streamsize>(sizeof(std::uint64_t)));
    }

    std::string cache_key_;
    const StreamedExpertTable& table_;
    std::filesystem::path path_;
    std::filesystem::path tmp_;
    std::ofstream os_;
    std::uint64_t header_bytes_ = 0;
    std::uint64_t body_cursor_ = 0;
};

bool ensure_gpt_oss_native_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose);

bool ensure_gpt_oss_eager_bf16_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose);

bool ensure_gpt_oss_routed_mxfp4_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose);

bool ensure_mixtral_tp_bf16_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose);

bool ensure_qwen35_moe_tp_bf16_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose);

bool ensure_qwen3_moe_tp_bf16_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose);

bool ensure_dsv4_tp_mxfp4_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose);

// Dispatch offline pack builders from `table.pack_kind` (set by the stream
// arch recipe). No-op when pack_kind is None.
inline void ensure_streamed_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose)
{
    using pie_weight_loader::PieLoaderExpertPackKind;
    switch (static_cast<PieLoaderExpertPackKind>(table.pack_kind)) {
    case PieLoaderExpertPackKind::None:
        return;
    case PieLoaderExpertPackKind::GptOssNativeMarlin:
        ensure_gpt_oss_native_expert_pack(
            table, cache_key, checkpoint, verbose);
        return;
    case PieLoaderExpertPackKind::GptOssEagerBf16:
        ensure_gpt_oss_eager_bf16_expert_pack(
            table, cache_key, checkpoint, verbose);
        return;
    case PieLoaderExpertPackKind::GptOssRoutedMxfp4:
        ensure_gpt_oss_routed_mxfp4_expert_pack(
            table, cache_key, checkpoint, verbose);
        return;
    case PieLoaderExpertPackKind::MixtralTpBf16:
        ensure_mixtral_tp_bf16_expert_pack(
            table, cache_key, checkpoint, verbose);
        return;
    case PieLoaderExpertPackKind::Qwen35MoeTpBf16:
        ensure_qwen35_moe_tp_bf16_expert_pack(
            table, cache_key, checkpoint, verbose);
        return;
    case PieLoaderExpertPackKind::Qwen3MoeTpBf16:
        ensure_qwen3_moe_tp_bf16_expert_pack(
            table, cache_key, checkpoint, verbose);
        return;
    case PieLoaderExpertPackKind::Dsv4TpMxfp4:
        ensure_dsv4_tp_mxfp4_expert_pack(
            table, cache_key, checkpoint, verbose);
        return;
    }
    throw std::runtime_error(
        "ensure_streamed_expert_pack: unknown pack_kind=" +
        std::to_string(table.pack_kind));
}

}  // namespace pie_cuda_driver
