#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "pipeline/page_translation.hpp"

int main() {
    const std::uint32_t translation[] = {10, 11};
    std::string error;

    std::vector<std::uint32_t> reads{0, 3};
    std::vector<std::uint32_t> writes{1};
    if (pie_cuda_driver::pipeline::translate_resolved_page_ids(
            reads, writes, translation, false, &error) ||
        error.find("unmasked read") == std::string::npos) {
        std::fputs("page_translation_test: unmasked read accepted\n", stderr);
        return 1;
    }

    reads = {0, 3};
    writes = {1};
    error.clear();
    if (!pie_cuda_driver::pipeline::translate_resolved_page_ids(
            reads, writes, translation, true, &error) ||
        reads != std::vector<std::uint32_t>({10, 0}) ||
        writes != std::vector<std::uint32_t>({11})) {
        std::fputs("page_translation_test: masked read fallback failed\n", stderr);
        return 1;
    }

    reads = {0};
    writes = {2};
    error.clear();
    if (pie_cuda_driver::pipeline::translate_resolved_page_ids(
            reads, writes, translation, true, &error) ||
        error.find("WSlot") == std::string::npos) {
        std::fputs("page_translation_test: invalid WSlot accepted\n", stderr);
        return 1;
    }

    std::puts("page_translation_test: OK");
    return 0;
}
