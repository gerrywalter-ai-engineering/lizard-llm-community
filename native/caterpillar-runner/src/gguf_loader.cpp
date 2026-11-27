// GGUF v2/v3 metadata and tensor-directory reader.
// Clean-room implementation derived from the public GGUF file-format
// specification. No upstream parser source was copied.

#include "gguf_loader.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace cat {
namespace {

enum class ValueType : std::uint32_t {
    uint8 = 0, int8 = 1, uint16 = 2, int16 = 3, uint32 = 4, int32 = 5,
    float32 = 6, boolean = 7, string = 8, array = 9, uint64 = 10,
    int64 = 11, float64 = 12,
};

class Reader {
public:
    explicit Reader(const std::filesystem::path& path)
        : stream_(path, std::ios::binary), size_(std::filesystem::file_size(path)) {
        if (!stream_) throw std::runtime_error("unable to open GGUF file: " + path.string());
    }

    template <typename T>
    T read() {
        static_assert(std::endian::native == std::endian::little, "Caterpillar currently requires a little-endian host");
        T value{};
        read_bytes(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }

    std::string read_string() {
        const auto length = read<std::uint64_t>();
        if (length > remaining() || length > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("invalid GGUF string length");
        }
        std::string value(static_cast<std::size_t>(length), '\0');
        if (length > 0) read_bytes(value.data(), static_cast<std::size_t>(length));
        return value;
    }

    void skip(std::uint64_t bytes) {
        if (bytes > remaining()) throw std::runtime_error("GGUF value extends past end of file");
        stream_.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
        if (!stream_) throw std::runtime_error("unable to seek through GGUF value");
    }

    [[nodiscard]] std::uint64_t position() {
        const auto value = stream_.tellg();
        if (value < 0) throw std::runtime_error("unable to query GGUF stream position");
        return static_cast<std::uint64_t>(value);
    }

    [[nodiscard]] std::uint64_t remaining() {
        const auto at = position();
        return at <= size_ ? size_ - at : 0;
    }

private:
    void read_bytes(char* destination, std::size_t bytes) {
        if (bytes > remaining()) throw std::runtime_error("truncated GGUF file");
        stream_.read(destination, static_cast<std::streamsize>(bytes));
        if (!stream_) throw std::runtime_error("unable to read GGUF data");
    }

    std::ifstream stream_;
    std::uint64_t size_{};
};

std::uint64_t scalar_width(ValueType type) {
    switch (type) {
        case ValueType::uint8:
        case ValueType::int8:
        case ValueType::boolean: return 1;
        case ValueType::uint16:
        case ValueType::int16: return 2;
        case ValueType::uint32:
        case ValueType::int32:
        case ValueType::float32: return 4;
        case ValueType::uint64:
        case ValueType::int64:
        case ValueType::float64: return 8;
        case ValueType::string:
        case ValueType::array: return 0;
    }
    throw std::runtime_error("unknown GGUF value type");
}

void skip_value(Reader& reader, ValueType type) {
    if (type == ValueType::string) {
        reader.skip(reader.read<std::uint64_t>());
        return;
    }
    if (type == ValueType::array) {
        const auto element_type = static_cast<ValueType>(reader.read<std::uint32_t>());
        const auto count = reader.read<std::uint64_t>();
        const auto width = scalar_width(element_type);
        if (width != 0) {
            if (count > std::numeric_limits<std::uint64_t>::max() / width) throw std::runtime_error("GGUF array size overflow");
            reader.skip(count * width);
        } else {
            for (std::uint64_t index = 0; index < count; ++index) skip_value(reader, element_type);
        }
        return;
    }
    reader.skip(scalar_width(type));
}

std::optional<std::uint64_t> read_unsigned(Reader& reader, ValueType type) {
    switch (type) {
        case ValueType::uint8: return reader.read<std::uint8_t>();
        case ValueType::uint16: return reader.read<std::uint16_t>();
        case ValueType::uint32: return reader.read<std::uint32_t>();
        case ValueType::uint64: return reader.read<std::uint64_t>();
        default: return std::nullopt;
    }
}

std::optional<double> read_number(Reader& reader, ValueType type) {
    if (const auto unsigned_value = read_unsigned(reader, type)) return static_cast<double>(*unsigned_value);
    switch (type) {
        case ValueType::int8: return reader.read<std::int8_t>();
        case ValueType::int16: return reader.read<std::int16_t>();
        case ValueType::int32: return reader.read<std::int32_t>();
        case ValueType::int64: return static_cast<double>(reader.read<std::int64_t>());
        case ValueType::float32: return reader.read<float>();
        case ValueType::float64: return reader.read<double>();
        default: return std::nullopt;
    }
}

std::optional<bool> read_boolean(Reader& reader, ValueType type) {
    if (type == ValueType::boolean) return reader.read<std::uint8_t>() != 0;
    if (const auto value = read_unsigned(reader, type)) return *value != 0;
    return std::nullopt;
}

std::optional<std::vector<std::string>> read_string_array(Reader& reader, ValueType type) {
    if (type != ValueType::array) return std::nullopt;
    const auto element_type = static_cast<ValueType>(reader.read<std::uint32_t>());
    const auto count = reader.read<std::uint64_t>();
    if (element_type != ValueType::string) {
        const auto width = scalar_width(element_type);
        if (width == 0) throw std::runtime_error("unsupported nested GGUF tokenizer array");
        if (count > std::numeric_limits<std::uint64_t>::max() / width) throw std::runtime_error("GGUF array size overflow");
        reader.skip(count * width);
        return std::nullopt;
    }
    if (count > 2'000'000U) throw std::runtime_error("GGUF tokenizer array is unreasonably large");
    std::vector<std::string> output;
    output.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) output.push_back(reader.read_string());
    return output;
}

template <typename Destination>
std::optional<std::vector<Destination>> read_numeric_array(Reader& reader, ValueType type) {
    if (type != ValueType::array) return std::nullopt;
    const auto element_type = static_cast<ValueType>(reader.read<std::uint32_t>());
    const auto count = reader.read<std::uint64_t>();
    if (count > 2'000'000U) throw std::runtime_error("GGUF tokenizer array is unreasonably large");
    std::vector<Destination> output;
    output.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        const auto value = read_number(reader, element_type);
        if (!value) {
            if (index == 0) {
                const auto width = scalar_width(element_type);
                if (width == 0) throw std::runtime_error("unsupported nested GGUF tokenizer array");
                reader.skip(count * width);
                return std::nullopt;
            }
            throw std::runtime_error("mixed or invalid GGUF numeric array");
        }
        output.push_back(static_cast<Destination>(*value));
    }
    return output;
}

std::string quant_name(std::uint64_t file_type) {
    switch (file_type) {
        case 0: return "F32"; case 1: return "F16"; case 2: return "Q4_0";
        case 3: return "Q4_1"; case 7: return "Q8_0"; case 8: return "Q5_0";
        case 9: return "Q5_1"; case 10: return "Q2_K"; case 11: return "Q3_K_S";
        case 12: return "Q3_K_M"; case 13: return "Q3_K_L"; case 14: return "Q4_K_S";
        case 15: return "Q4_K_M"; case 16: return "Q5_K_S"; case 17: return "Q5_K_M";
        case 18: return "Q6_K"; case 25: return "IQ4_NL"; case 30: return "IQ4_XS";
        case 32: return "BF16"; case 36: return "TQ1_0"; case 37: return "TQ2_0";
        default: return "UNKNOWN";
    }
}

std::string family_name(std::string architecture) {
    std::transform(architecture.begin(), architecture.end(), architecture.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    if (architecture == "gemma" || architecture == "gemma2" || architecture == "gemma3" || architecture == "gemma4") return "gemma";
    if (architecture == "qwen2" || architecture == "qwen3") return architecture;
    if (architecture == "mistral") return "llama";
    return architecture;
}

std::uint64_t align_up(std::uint64_t value, std::uint32_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) throw std::runtime_error("invalid GGUF alignment");
    return (value + alignment - 1) & ~static_cast<std::uint64_t>(alignment - 1);
}

std::string escape_json(std::string_view value) {
    std::ostringstream output;
    for (const char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << character; break;
        }
    }
    return output.str();
}

template <typename T>
void write_scalar(std::ofstream& stream, T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_string(std::ofstream& stream, std::string_view value) {
    write_scalar(stream, static_cast<std::uint64_t>(value.size()));
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

}  // namespace

GgufModelInfo inspect_gguf(const std::filesystem::path& path) {
    Reader reader(path);
    char magic[4]{};
    for (char& value : magic) value = reader.read<char>();
    if (std::string_view(magic, 4) != "GGUF") throw std::runtime_error("not a GGUF file");
    GgufModelInfo info;
    info.version = reader.read<std::uint32_t>();
    if (info.version < 2 || info.version > 3) throw std::runtime_error("unsupported GGUF version");
    const auto tensor_count = reader.read<std::uint64_t>();
    info.metadata_count = reader.read<std::uint64_t>();

    // Architecture-scoped keys are matched by suffix (".embedding_length" and
    // friends), because the prefix is the architecture name and varies per
    // model. A multimodal GGUF also carries a submodel under its own namespace
    // - gemma3 ships gemma3.vision.embedding_length beside
    // gemma3.embedding_length - and those suffixes match just as well. Read in
    // file order the vision value landed last and won, so a gemma3-4b decoder
    // was built with the vision tower's 1152 hidden size and 27 blocks, and
    // load died on "GGUF vector shape differs from graph parameter:
    // blk.0.attn_norm.weight" against a [2560] tensor.
    //
    // The text decoder is what this runner executes, so submodel namespaces are
    // skipped outright rather than parsed and discarded field by field.
    const auto is_submodel_key = [](const std::string& name) {
        return name.find(".vision.") != std::string::npos
            || name.find(".clip.") != std::string::npos
            || name.find(".audio.") != std::string::npos;
    };
    for (std::uint64_t index = 0; index < info.metadata_count; ++index) {
        const auto key = reader.read_string();
        const auto type = static_cast<ValueType>(reader.read<std::uint32_t>());
        if (is_submodel_key(key)) {
            skip_value(reader, type);
            continue;
        }
        if ((key == "general.architecture" || key == "general.name") && type == ValueType::string) {
            const auto value = reader.read_string();
            if (key == "general.architecture") info.architecture = value;
            else info.name = value;
        } else if (key == "tokenizer.ggml.model" && type == ValueType::string) {
            info.tokenizer_model = reader.read_string();
        } else if (key == "tokenizer.ggml.tokens") {
            const auto value = read_string_array(reader, type);
            if (value) info.token_vocabulary = std::move(*value); else if (type != ValueType::array) skip_value(reader, type);
        } else if (key == "tokenizer.ggml.scores") {
            const auto value = read_numeric_array<float>(reader, type);
            if (value) info.token_scores = std::move(*value); else if (type != ValueType::array) skip_value(reader, type);
        } else if (key == "tokenizer.ggml.token_type") {
            const auto value = read_numeric_array<std::int32_t>(reader, type);
            if (value) info.token_types = std::move(*value); else if (type != ValueType::array) skip_value(reader, type);
        } else if (key == "tokenizer.ggml.merges") {
            const auto value = read_string_array(reader, type);
            if (value) info.token_merges = std::move(*value); else if (type != ValueType::array) skip_value(reader, type);
        } else if (key == "tokenizer.ggml.bos_token_id") {
            const auto value = read_unsigned(reader, type);
            if (value) info.bos_token_id = *value; else skip_value(reader, type);
        } else if (key == "tokenizer.ggml.eos_token_id") {
            const auto value = read_unsigned(reader, type);
            if (value) info.eos_token_id = *value; else skip_value(reader, type);
        } else if (key == "tokenizer.ggml.unknown_token_id") {
            const auto value = read_unsigned(reader, type);
            if (value) info.unknown_token_id = *value; else skip_value(reader, type);
        } else if (key == "tokenizer.ggml.padding_token_id") {
            const auto value = read_unsigned(reader, type);
            if (value) info.padding_token_id = *value; else skip_value(reader, type);
        } else if (key == "tokenizer.ggml.add_bos_token") {
            const auto value = read_boolean(reader, type);
            if (value) info.add_bos_token = *value; else skip_value(reader, type);
        } else if (key == "tokenizer.ggml.add_eos_token") {
            const auto value = read_boolean(reader, type);
            if (value) info.add_eos_token = *value; else skip_value(reader, type);
        } else if (key == "tokenizer.ggml.add_space_prefix") {
            const auto value = read_boolean(reader, type);
            if (value) info.add_space_prefix = *value; else skip_value(reader, type);
        } else if (key == "general.file_type") {
            const auto value = read_unsigned(reader, type);
            if (value) info.quantization = quant_name(*value); else skip_value(reader, type);
        } else if (key == "general.alignment") {
            const auto value = read_unsigned(reader, type);
            if (value && *value <= std::numeric_limits<std::uint32_t>::max()) info.alignment = static_cast<std::uint32_t>(*value);
            else if (!value) skip_value(reader, type);
        } else if (key.ends_with(".block_count")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.block_count = *value; else skip_value(reader, type);
        } else if (key.ends_with(".embedding_length")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.embedding_length = *value; else skip_value(reader, type);
        } else if (key.ends_with(".feed_forward_length")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.feed_forward_length = *value; else skip_value(reader, type);
        } else if (key.ends_with(".attention.head_count")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.head_count = *value; else skip_value(reader, type);
        } else if (key.ends_with(".attention.head_count_kv")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.head_count_kv = *value; else skip_value(reader, type);
        } else if (key.ends_with(".attention.key_length")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.key_length = *value; else skip_value(reader, type);
        } else if (key.ends_with(".attention.value_length")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.value_length = *value; else skip_value(reader, type);
        } else if (key.ends_with(".rope.dimension_count")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.rope_dimension_count = *value; else skip_value(reader, type);
        } else if (key.ends_with(".context_length")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.context_length = *value; else skip_value(reader, type);
        } else if (key.ends_with(".attention.key_length_swa")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.attention_key_length_swa = *value; else skip_value(reader, type);
        } else if (key.ends_with(".attention.value_length_swa")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.attention_value_length_swa = *value; else skip_value(reader, type);
        } else if (key.ends_with(".attention.key_length")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.attention_key_length = *value; else skip_value(reader, type);
        } else if (key.ends_with(".attention.value_length")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.attention_value_length = *value; else skip_value(reader, type);
        } else if (key.ends_with(".attention.sliding_window")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.attention_sliding_window = *value; else skip_value(reader, type);
        } else if (key.ends_with(".attention.shared_kv_layers")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.attention_shared_kv_layers = *value; else skip_value(reader, type);
        } else if (key.ends_with(".embedding_length_per_layer_input")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.embedding_length_per_layer_input = *value; else skip_value(reader, type);
        } else if (key.ends_with(".vocab_size")) {
            const auto value = read_unsigned(reader, type);
            if (value) info.vocabulary_size = *value; else skip_value(reader, type);
        } else if (key.ends_with(".rope.freq_base")) {
            const auto value = read_number(reader, type);
            if (value) info.rope_frequency_base = static_cast<float>(*value); else skip_value(reader, type);
        } else if (key.ends_with(".rope.freq_base_swa")) {
            const auto value = read_number(reader, type);
            if (value) info.rope_frequency_base_swa = static_cast<float>(*value); else skip_value(reader, type);
        } else if (key.ends_with(".rope.scale_linear") || key.ends_with(".rope.freq_scale_train")) {
            const auto value = read_number(reader, type);
            if (value) info.rope_frequency_scale = static_cast<float>(*value); else skip_value(reader, type);
        } else if (key.ends_with(".final_logit_softcapping")) {
            const auto value = read_number(reader, type);
            if (value) info.final_logit_softcap = static_cast<float>(*value); else skip_value(reader, type);
        } else if (key.ends_with(".attention.layer_norm_rms_epsilon")) {
            const auto value = read_number(reader, type);
            if (value) info.rms_epsilon = static_cast<float>(*value); else skip_value(reader, type);
        } else {
            skip_value(reader, type);
        }
    }

    info.tensors.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t index = 0; index < tensor_count; ++index) {
        GgufTensorInfo tensor;
        tensor.name = reader.read_string();
        const auto dimensions = reader.read<std::uint32_t>();
        if (dimensions == 0 || dimensions > 8) throw std::runtime_error("invalid GGUF tensor rank");
        tensor.dimensions.reserve(dimensions);
        for (std::uint32_t dimension = 0; dimension < dimensions; ++dimension) {
            tensor.dimensions.push_back(reader.read<std::uint64_t>());
        }
        tensor.type = reader.read<std::uint32_t>();
        tensor.offset = reader.read<std::uint64_t>();
        info.tensors.push_back(std::move(tensor));
    }
    info.tensor_data_offset = align_up(reader.position(), info.alignment);
    info.family = family_name(info.architecture);
    if (!info.token_vocabulary.empty()) info.vocabulary_size = info.token_vocabulary.size();

    if (info.family == "gemma" && info.block_count > 0) {
        std::unordered_set<std::string> names;
        for (const auto& tensor : info.tensors) names.insert(tensor.name);
        for (std::uint64_t layer = 0; layer < info.block_count; ++layer) {
            if (!names.contains("blk." + std::to_string(layer) + ".attn_v.weight")) {
                info.missing_attention_value_layers.push_back(layer);
            }
        }
    }
    return info;
}

std::string gguf_info_json(const GgufModelInfo& info) {
    std::ostringstream output;
    output << "{\"version\":" << info.version
           << ",\"architecture\":\"" << escape_json(info.architecture)
           << "\",\"family\":\"" << escape_json(info.family)
           << "\",\"quantization\":\"" << escape_json(info.quantization)
           << "\",\"blockCount\":" << info.block_count
           << ",\"embeddingLength\":" << info.embedding_length
           << ",\"feedForwardLength\":" << info.feed_forward_length
           << ",\"headCount\":" << info.head_count
           << ",\"headCountKv\":" << info.head_count_kv
           << ",\"ropeDimensionCount\":" << info.rope_dimension_count
           << ",\"contextLength\":" << info.context_length
           << ",\"attentionKeyLength\":" << info.attention_key_length
           << ",\"attentionKeyLengthSwa\":" << info.attention_key_length_swa
           << ",\"attentionSlidingWindow\":" << info.attention_sliding_window
           << ",\"sharedKvLayers\":" << info.attention_shared_kv_layers
           << ",\"perLayerInputLength\":" << info.embedding_length_per_layer_input
           << ",\"finalLogitSoftcap\":" << info.final_logit_softcap
           << ",\"vocabularySize\":" << info.vocabulary_size
           << ",\"tokenizerModel\":\"" << escape_json(info.tokenizer_model)
           << "\",\"tokenMergeCount\":" << info.token_merges.size()
           << ",\"bosTokenId\":" << info.bos_token_id
           << ",\"eosTokenId\":" << info.eos_token_id
           << ",\"addSpacePrefix\":" << (info.add_space_prefix ? "true" : "false")
           << ",\"tensorCount\":" << info.tensors.size()
           << ",\"tensorDataOffset\":" << info.tensor_data_offset
           << ",\"missingAttentionValueLayers\":[";
    for (std::size_t index = 0; index < info.missing_attention_value_layers.size(); ++index) {
        if (index != 0) output << ',';
        output << info.missing_attention_value_layers[index];
    }
    output << "]}";
    return output.str();
}

int run_gguf_selftest() {
    const auto path = std::filesystem::temp_directory_path() / "caterpillar-gguf-selftest.gguf";
    try {
        {
            std::ofstream stream(path, std::ios::binary | std::ios::trunc);
            stream.write("GGUF", 4);
            write_scalar(stream, std::uint32_t{3});
            write_scalar(stream, std::uint64_t{2});
            write_scalar(stream, std::uint64_t{4});
            write_string(stream, "general.architecture"); write_scalar(stream, std::uint32_t{8}); write_string(stream, "gemma4");
            write_string(stream, "general.file_type"); write_scalar(stream, std::uint32_t{4}); write_scalar(stream, std::uint32_t{15});
            write_string(stream, "gemma4.block_count"); write_scalar(stream, std::uint32_t{4}); write_scalar(stream, std::uint32_t{2});
            write_string(stream, "gemma4.embedding_length"); write_scalar(stream, std::uint32_t{4}); write_scalar(stream, std::uint32_t{2560});
            for (std::uint64_t layer = 0; layer < 2; ++layer) {
                write_string(stream, "blk." + std::to_string(layer) + ".attn_v.weight");
                write_scalar(stream, std::uint32_t{2});
                write_scalar(stream, std::uint64_t{4}); write_scalar(stream, std::uint64_t{4});
                write_scalar(stream, std::uint32_t{0}); write_scalar(stream, layer * 64);
            }
        }
        const auto info = inspect_gguf(path);
        std::filesystem::remove(path);
        if (info.version != 3 || info.architecture != "gemma4" || info.family != "gemma"
            || info.quantization != "Q4_K_M" || info.block_count != 2
            || info.embedding_length != 2560 || info.tensors.size() != 2
            || !info.missing_attention_value_layers.empty()) {
            throw std::runtime_error("parsed GGUF metadata did not match the fixture");
        }
        std::cout << "caterpillar GGUF loader selftest: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << "caterpillar GGUF loader selftest: failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace cat
