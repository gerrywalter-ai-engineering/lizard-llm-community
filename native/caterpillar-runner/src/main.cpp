// Caterpillar standalone inference runner.
// Clean-room implementation: no llama.cpp, ggml, ik_llama.cpp, or
// lizard-native translation unit is included or copied here.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>
#include <vector>

#include "selftest.h"
#include "backend_d3d12.h"
#include "gguf_loader.h"
#include "tensor_data.h"
#include "mapped_model.h"
#include "cpu_model.h"
#include "tokenizer.h"
#include "family_spec.h"
#include "fit.h"
#include "plan.h"
#include "server.h"
#include "kv_cache.h"
#include "spec.h"

int main(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "--selftest") {
        std::cout << "caterpillar-runner selftest: ok\n";
        return EXIT_SUCCESS;
    }
    if (argc > 1 && std::string_view(argv[1]) == "--graph-selftest") {
        return cat::run_graph_selftest();
    }
    if (argc > 1 && std::string_view(argv[1]) == "--gguf-selftest") {
        return cat::run_gguf_selftest();
    }
    if (argc > 1 && std::string_view(argv[1]) == "--tensor-selftest") {
        return cat::run_tensor_data_selftest();
    }
    if (argc > 1 && std::string_view(argv[1]) == "--mapped-model-selftest") {
        return cat::run_mapped_model_selftest();
    }
    if (argc > 1 && std::string_view(argv[1]) == "--tokenizer-selftest") {
        return cat::run_tokenizer_selftest();
    }
    if (argc > 1 && std::string_view(argv[1]) == "--family-graph-selftest") {
        return cat::run_family_graph_selftest();
    }
    if (argc > 1 && std::string_view(argv[1]) == "--fit-selftest") {
        return cat::run_fit_selftest();
    }
    if (argc > 1 && std::string_view(argv[1]) == "--gpu-selftest") {
        return cat::run_gpu_selftest();
    }
    if (argc > 1 && std::string_view(argv[1]) == "--kv-cache-selftest") {
        return cat::run_kv_cache_selftest();
    }
    if (argc > 1 && std::string_view(argv[1]) == "--spec-selftest") {
        return cat::run_spec_selftest();
    }
    if (argc > 2 && std::string_view(argv[1]) == "--inspect-gguf") {
        try {
            std::cout << cat::gguf_info_json(cat::inspect_gguf(argv[2])) << '\n';
            return EXIT_SUCCESS;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
    if (argc > 3 && std::string_view(argv[1]) == "--graph-summary-gguf") {
        try {
            const auto tokens = static_cast<std::size_t>(std::stoull(argv[3]));
            const auto family = cat::resolve_family_spec(cat::inspect_gguf(argv[2]));
            const auto decoder = cat::build_decoder_graph(family, tokens);
            const auto plan = cat::Plan::compile(decoder.graph);
            std::cout << "{\"architecture\":\"" << family.architecture
                      << "\",\"family\":\"" << family.family
                      << "\",\"layers\":" << family.layers
                      << ",\"tokens\":" << tokens
                      << ",\"nodes\":" << decoder.graph.nodes().size()
                      << ",\"cpuSpans\":" << plan.cpu_spans().size()
                      << ",\"activationArenaBytes\":" << plan.allocation().arena_bytes
                      << ",\"naiveActivationBytes\":" << plan.allocation().naive_bytes
                      << "}\n";
            return EXIT_SUCCESS;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
    if (argc > 3 && std::string_view(argv[1]) == "--tensor-summary-gguf") {
        try {
            const auto model = cat::inspect_gguf(argv[2]);
            const auto values = cat::read_tensor_f32(argv[2], model, argv[3]);
            const auto tensor_info = std::find_if(model.tensors.begin(), model.tensors.end(), [&](const cat::GgufTensorInfo& tensor) {
                return tensor.name == argv[3];
            });
            if (tensor_info == model.tensors.end()) throw std::runtime_error("tensor is absent from GGUF directory");
            if (values.empty()) throw std::runtime_error("decoded tensor is empty");
            float minimum = values.front();
            float maximum = values.front();
            long double checksum = 0.0L;
            for (std::size_t index = 0; index < values.size(); ++index) {
                if (!std::isfinite(values[index])) throw std::runtime_error("decoded tensor contains a non-finite value");
                minimum = std::min(minimum, values[index]);
                maximum = std::max(maximum, values[index]);
                checksum += static_cast<long double>(values[index]) * static_cast<long double>((index % 251U) + 1U);
            }
            std::cout << "{\"tensor\":\"" << argv[3]
                      << "\",\"ggmlType\":" << tensor_info->type
                      << ",\"elements\":" << values.size()
                      << ",\"minimum\":" << minimum
                      << ",\"maximum\":" << maximum
                      << ",\"weightedChecksum\":" << static_cast<double>(checksum)
                      << ",\"first\":[";
            for (std::size_t index = 0; index < std::min<std::size_t>(8, values.size()); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << values[index];
            }
            std::cout << ']'
                      << "}\n";
            return EXIT_SUCCESS;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
    if (argc > 3 && std::string_view(argv[1]) == "--tensor-info-gguf") {
        try {
            const auto model = cat::inspect_gguf(argv[2]);
            const auto found = std::find_if(model.tensors.begin(), model.tensors.end(), [&](const cat::GgufTensorInfo& tensor) {
                return tensor.name == argv[3];
            });
            if (found == model.tensors.end()) throw std::runtime_error("tensor is absent from GGUF directory");
            std::cout << "{\"tensor\":\"" << found->name << "\",\"ggmlType\":" << found->type << ",\"dimensions\":[";
            for (std::size_t index = 0; index < found->dimensions.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << found->dimensions[index];
            }
            std::cout << "]}\n";
            return EXIT_SUCCESS;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
    if (argc > 3 && std::string_view(argv[1]) == "--tensor-list-gguf") {
        try {
            const auto model = cat::inspect_gguf(argv[2]);
            const std::string_view prefix = argv[3];
            for (const auto& tensor : model.tensors) {
                if (!std::string_view(tensor.name).starts_with(prefix)) continue;
                std::cout << tensor.name << " type=" << tensor.type << " shape=";
                for (std::size_t index = 0; index < tensor.dimensions.size(); ++index) {
                    if (index != 0) std::cout << 'x';
                    std::cout << tensor.dimensions[index];
                }
                std::cout << '\n';
            }
            return EXIT_SUCCESS;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
    if (argc > 3 && std::string_view(argv[1]) == "--tokenize-gguf") {
        try {
            const auto model = cat::inspect_gguf(argv[2]);
            const cat::Tokenizer tokenizer(model);
            const auto tokens = tokenizer.encode(argv[3]);
            std::cout << "{\"tokenizerModel\":\"" << model.tokenizer_model << "\",\"tokens\":[";
            for (std::size_t index = 0; index < tokens.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << tokens[index];
            }
            std::cout << "],\"pieces\":[";
            for (std::size_t index = 0; index < tokens.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << '"';
                for (const char value : tokenizer.token_text(tokens[index])) {
                    if (value == '"' || value == '\\') std::cout << '\\';
                    std::cout << value;
                }
                std::cout << '"';
            }
            std::cout << "],\"decoded\":\"";
            const auto decoded = tokenizer.decode(tokens);
            for (const char value : decoded) {
                if (value == '"' || value == '\\') std::cout << '\\';
                std::cout << value;
            }
            std::cout << "\"}\n";
            return EXIT_SUCCESS;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
    if (argc > 3 && std::string_view(argv[1]) == "--cpu-greedy-gguf") {
        try {
            const std::size_t max_tokens = argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : 1U;
            const cat::CpuModel model(argv[2]);
            const auto result = model.generate(argv[3], max_tokens);
            std::cout << "{\"promptTokens\":[";
            for (std::size_t index = 0; index < result.prompt_tokens.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << result.prompt_tokens[index];
            }
            std::cout << "],\"generatedTokens\":[";
            for (std::size_t index = 0; index < result.generated_tokens.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << result.generated_tokens[index];
            }
            std::cout << "],\"text\":\"";
            for (const char value : result.text) {
                if (value == '"' || value == '\\') std::cout << '\\';
                if (value == '\n') std::cout << "\\n";
                else if (value == '\r') std::cout << "\\r";
                else std::cout << value;
            }
            std::cout << "\"}\n";
            return EXIT_SUCCESS;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
    if (argc > 3 && std::string_view(argv[1]) == "--cpu-greedy-tokens-gguf") {
        try {
            std::vector<std::uint32_t> tokens;
            std::istringstream input(argv[3]);
            std::string item;
            while (std::getline(input, item, ',')) {
                if (item.empty()) throw std::invalid_argument("token list contains an empty item");
                tokens.push_back(static_cast<std::uint32_t>(std::stoul(item)));
            }
            if (tokens.empty()) throw std::invalid_argument("token list is empty");
            const cat::CpuModel model(argv[2]);
            const auto layer_limit = argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : 0U;
            const auto generated = model.greedy_next(tokens, layer_limit);
            std::cout << "{\"promptTokens\":[";
            for (std::size_t index = 0; index < tokens.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << tokens[index];
            }
            std::cout << "],\"generatedToken\":" << generated
                      << ",\"text\":\"" << model.tokenizer().decode({generated}) << "\"}\n";
            return EXIT_SUCCESS;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
    if (argc > 3 && std::string_view(argv[1]) == "--hidden-summary-tokens-gguf") {
        try {
            std::vector<std::uint32_t> tokens;
            std::istringstream input(argv[3]);
            std::string item;
            while (std::getline(input, item, ',')) tokens.push_back(static_cast<std::uint32_t>(std::stoul(item)));
            if (tokens.empty()) throw std::invalid_argument("token list is empty");
            const cat::CpuModel model(argv[2]);
            const auto layer_limit = argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : 0U;
            const auto states = model.hidden_states(tokens, layer_limit);
            const auto width = static_cast<std::size_t>(model.info().embedding_length);
            const auto offset = states.size() - width;
            double sum = 0.0;
            double square_sum = 0.0;
            double weighted = 0.0;
            float minimum = states[offset];
            float maximum = states[offset];
            for (std::size_t index = 0; index < width; ++index) {
                const float value = states[offset + index];
                sum += value;
                square_sum += static_cast<double>(value) * value;
                weighted += static_cast<double>(value) * static_cast<double>(1 + index % 251);
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
            std::cout << "{\"count\":" << width << ",\"sum\":" << sum
                      << ",\"rms\":" << std::sqrt(square_sum / static_cast<double>(width))
                      << ",\"weightedChecksum\":" << weighted << ",\"min\":" << minimum
                      << ",\"max\":" << maximum << ",\"first\":[";
            for (std::size_t index = 0; index < std::min<std::size_t>(8, width); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << states[offset + index];
            }
            std::cout << "]}\n";
            return EXIT_SUCCESS;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
    return cat::run_rpc_server();
}
