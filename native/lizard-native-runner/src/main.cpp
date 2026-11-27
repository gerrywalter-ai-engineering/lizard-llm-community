#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <atomic>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef LIZARD_NATIVE_AVX2
#include <immintrin.h>
#endif

#include "d3d12_batch_matmul.h"
#include "engine_loop.h"
#include "ggml_cpu_backend.h"
#include "kv_blocks.h"
#include "prefix_cache.h"
#include "hardware_probe.h"

namespace lizard_native {

struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue {
  using Variant = std::variant<std::nullptr_t, bool, double, std::string, JsonObject, JsonArray>;

  Variant value;

  JsonValue() : value(nullptr) {}
  JsonValue(std::nullptr_t) : value(nullptr) {}
  JsonValue(bool v) : value(v) {}
  JsonValue(double v) : value(v) {}
  JsonValue(int v) : value(static_cast<double>(v)) {}
  JsonValue(std::uint32_t v) : value(static_cast<double>(v)) {}
  JsonValue(std::uint64_t v) : value(static_cast<double>(v)) {}
  JsonValue(const char *v) : value(std::string(v)) {}
  JsonValue(std::string v) : value(std::move(v)) {}
  JsonValue(JsonObject v) : value(std::move(v)) {}
  JsonValue(JsonArray v) : value(std::move(v)) {}

  bool is_null() const { return std::holds_alternative<std::nullptr_t>(value); }
  bool is_bool() const { return std::holds_alternative<bool>(value); }
  bool is_number() const { return std::holds_alternative<double>(value); }
  bool is_string() const { return std::holds_alternative<std::string>(value); }
  bool is_object() const { return std::holds_alternative<JsonObject>(value); }
  bool is_array() const { return std::holds_alternative<JsonArray>(value); }

  bool as_bool() const { return std::get<bool>(value); }
  double as_number() const { return std::get<double>(value); }
  const std::string &as_string() const { return std::get<std::string>(value); }
  const JsonObject &as_object() const { return std::get<JsonObject>(value); }
  const JsonArray &as_array() const { return std::get<JsonArray>(value); }

  JsonObject &as_object() { return std::get<JsonObject>(value); }
  JsonArray &as_array() { return std::get<JsonArray>(value); }
};

static std::string trim(const std::string &value) {
  std::size_t start = 0;
  while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
    start += 1;
  }
  std::size_t end = value.size();
  while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    end -= 1;
  }
  return value.substr(start, end - start);
}

static std::string to_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

static std::string to_upper(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return value;
}

static bool ends_with_ci(const std::string &value, const std::string &suffix) {
  if (suffix.size() > value.size()) {
    return false;
  }
  return to_lower(value.substr(value.size() - suffix.size())) == to_lower(suffix);
}

static std::string escape_json_string(const std::string &value) {
  std::ostringstream output;
  for (unsigned char c : value) {
    switch (c) {
      case '\"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (c < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
        } else {
          output << static_cast<char>(c);
        }
        break;
    }
  }
  return output.str();
}

static std::string stringify_json(const JsonValue &value);

static std::string stringify_number(double value) {
  if (value == static_cast<double>(static_cast<std::int64_t>(value))) {
    std::ostringstream output;
    output << static_cast<std::int64_t>(value);
    return output.str();
  }
  std::ostringstream output;
  output << std::setprecision(15) << value;
  return output.str();
}

static std::string stringify_json(const JsonValue &value) {
  if (value.is_null()) {
    return "null";
  }
  if (value.is_bool()) {
    return value.as_bool() ? "true" : "false";
  }
  if (value.is_number()) {
    return stringify_number(value.as_number());
  }
  if (value.is_string()) {
    return "\"" + escape_json_string(value.as_string()) + "\"";
  }
  if (value.is_array()) {
    std::ostringstream output;
    output << "[";
    bool first = true;
    for (const JsonValue &entry : value.as_array()) {
      if (!first) {
        output << ",";
      }
      first = false;
      output << stringify_json(entry);
    }
    output << "]";
    return output.str();
  }

  std::ostringstream output;
  output << "{";
  bool first = true;
  for (const auto &[key, entry] : value.as_object()) {
    if (!first) {
      output << ",";
    }
    first = false;
    output << "\"" << escape_json_string(key) << "\":" << stringify_json(entry);
  }
  output << "}";
  return output.str();
}

static void append_utf8_from_code_point(std::string &out, std::uint32_t code_point) {
  if (code_point <= 0x7F) {
    out.push_back(static_cast<char>(code_point));
  } else if (code_point <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else if (code_point <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  }
}

class JsonParser {
 public:
  explicit JsonParser(std::string text) : text_(std::move(text)), offset_(0) {}

  JsonValue parse() {
    skip_whitespace();
    JsonValue value = parse_value();
    skip_whitespace();
    if (offset_ != text_.size()) {
      throw std::runtime_error("Unexpected trailing JSON input.");
    }
    return value;
  }

 private:
  std::string text_;
  std::size_t offset_;

  void skip_whitespace() {
    while (offset_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[offset_])) != 0) {
      offset_ += 1;
    }
  }

  char peek() const {
    if (offset_ >= text_.size()) {
      throw std::runtime_error("Unexpected end of JSON input.");
    }
    return text_[offset_];
  }

  char consume() {
    char c = peek();
    offset_ += 1;
    return c;
  }

  void expect(const std::string &literal) {
    if (text_.substr(offset_, literal.size()) != literal) {
      throw std::runtime_error("Invalid JSON literal.");
    }
    offset_ += literal.size();
  }

  JsonValue parse_value() {
    skip_whitespace();
    char c = peek();
    if (c == '{') {
      return JsonValue(parse_object());
    }
    if (c == '[') {
      return JsonValue(parse_array());
    }
    if (c == '"') {
      return JsonValue(parse_string());
    }
    if (c == 't') {
      expect("true");
      return JsonValue(true);
    }
    if (c == 'f') {
      expect("false");
      return JsonValue(false);
    }
    if (c == 'n') {
      expect("null");
      return JsonValue();
    }
    return JsonValue(parse_number());
  }

  JsonObject parse_object() {
    JsonObject output;
    consume();
    skip_whitespace();
    if (peek() == '}') {
      consume();
      return output;
    }
    while (true) {
      skip_whitespace();
      std::string key = parse_string();
      skip_whitespace();
      if (consume() != ':') {
        throw std::runtime_error("Expected ':' in JSON object.");
      }
      skip_whitespace();
      output.emplace(std::move(key), parse_value());
      skip_whitespace();
      char next = consume();
      if (next == '}') {
        break;
      }
      if (next != ',') {
        throw std::runtime_error("Expected ',' in JSON object.");
      }
    }
    return output;
  }

  JsonArray parse_array() {
    JsonArray output;
    consume();
    skip_whitespace();
    if (peek() == ']') {
      consume();
      return output;
    }
    while (true) {
      skip_whitespace();
      output.push_back(parse_value());
      skip_whitespace();
      char next = consume();
      if (next == ']') {
        break;
      }
      if (next != ',') {
        throw std::runtime_error("Expected ',' in JSON array.");
      }
    }
    return output;
  }

  std::string parse_string() {
    if (consume() != '"') {
      throw std::runtime_error("Expected JSON string.");
    }
    std::string output;
    while (true) {
      if (offset_ >= text_.size()) {
        throw std::runtime_error("Unterminated JSON string.");
      }
      char c = consume();
      if (c == '"') {
        break;
      }
      if (c != '\\') {
        output.push_back(c);
        continue;
      }
      char escaped = consume();
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          output.push_back(escaped);
          break;
        case 'b':
          output.push_back('\b');
          break;
        case 'f':
          output.push_back('\f');
          break;
        case 'n':
          output.push_back('\n');
          break;
        case 'r':
          output.push_back('\r');
          break;
        case 't':
          output.push_back('\t');
          break;
        case 'u': {
          if (offset_ + 4 > text_.size()) {
            throw std::runtime_error("Invalid JSON unicode escape.");
          }
          std::uint32_t code_point = 0;
          for (int index = 0; index < 4; index += 1) {
            char hex = text_[offset_ + index];
            code_point <<= 4;
            if (hex >= '0' && hex <= '9') {
              code_point += static_cast<std::uint32_t>(hex - '0');
            } else if (hex >= 'A' && hex <= 'F') {
              code_point += static_cast<std::uint32_t>(hex - 'A' + 10);
            } else if (hex >= 'a' && hex <= 'f') {
              code_point += static_cast<std::uint32_t>(hex - 'a' + 10);
            } else {
              throw std::runtime_error("Invalid JSON unicode escape.");
            }
          }
          offset_ += 4;
          append_utf8_from_code_point(output, code_point);
          break;
        }
        default:
          throw std::runtime_error("Unsupported JSON escape sequence.");
      }
    }
    return output;
  }

  double parse_number() {
    std::size_t start = offset_;
    if (peek() == '-') {
      offset_ += 1;
    }
    const std::size_t integer_start = offset_;
    while (offset_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[offset_])) != 0) {
      offset_ += 1;
    }
    if (offset_ == integer_start) {
      throw std::runtime_error("Invalid JSON number.");
    }
    if (offset_ < text_.size() && text_[offset_] == '.') {
      offset_ += 1;
      const std::size_t fraction_start = offset_;
      while (offset_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[offset_])) != 0) {
        offset_ += 1;
      }
      if (offset_ == fraction_start) {
        throw std::runtime_error("Invalid JSON number.");
      }
    }
    if (offset_ < text_.size() && (text_[offset_] == 'e' || text_[offset_] == 'E')) {
      offset_ += 1;
      if (offset_ < text_.size() && (text_[offset_] == '+' || text_[offset_] == '-')) {
        offset_ += 1;
      }
      const std::size_t exponent_start = offset_;
      while (offset_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[offset_])) != 0) {
        offset_ += 1;
      }
      if (offset_ == exponent_start) {
        throw std::runtime_error("Invalid JSON number.");
      }
    }
    return std::stod(text_.substr(start, offset_ - start));
  }
};

static const JsonValue *find_object_value(const JsonObject &object, const std::string &key) {
  auto iterator = object.find(key);
  return iterator == object.end() ? nullptr : &iterator->second;
}

static const JsonObject *find_object_field(const JsonObject &object, const std::string &key) {
  const JsonValue *value = find_object_value(object, key);
  if (value == nullptr || !value->is_object()) {
    return nullptr;
  }
  return &value->as_object();
}

static std::string json_string_or_default(const JsonObject &object, const std::string &key, const std::string &fallback = "") {
  const JsonValue *value = find_object_value(object, key);
  return value != nullptr && value->is_string() ? value->as_string() : fallback;
}

static bool json_bool_or_default(const JsonObject &object, const std::string &key, bool fallback) {
  const JsonValue *value = find_object_value(object, key);
  return value != nullptr && value->is_bool() ? value->as_bool() : fallback;
}

static double json_number_or_default(const JsonObject &object, const std::string &key, double fallback) {
  const JsonValue *value = find_object_value(object, key);
  return value != nullptr && value->is_number() ? value->as_number() : fallback;
}

class WorkerError : public std::runtime_error {
 public:
  WorkerError(std::string code, std::string message, JsonValue details = JsonValue())
    : std::runtime_error(std::move(message)), code_(std::move(code)), details_(std::move(details)) {}

  const std::string &code() const {
    return code_;
  }

  const JsonValue &details() const {
    return details_;
  }

 private:
  std::string code_;
  JsonValue details_;
};

class FileView {
 public:
  FileView() = default;
  ~FileView() {
    close();
  }

  FileView(const FileView &) = delete;
  FileView &operator=(const FileView &) = delete;

  void open(const std::filesystem::path &file_path, bool use_mmap, bool pin_memory) {
    close();
    file_path_ = file_path;
    pin_requested_ = pin_memory;
    file_size_ = static_cast<std::size_t>(std::filesystem::file_size(file_path_));
    if (use_mmap) {
      try {
        open_mapped();
        mode_ = "mmap";
        hint_sequential_prefetch();
      } catch (const std::exception &error) {
        const std::string warning = std::string("mmap unavailable: ") + error.what() + ". Falling back to buffered file reads.";
        close();
        file_path_ = file_path;
        pin_requested_ = pin_memory;
        file_size_ = static_cast<std::size_t>(std::filesystem::file_size(file_path_));
        warnings_.push_back(warning);
        open_buffered();
        mode_ = "no-mmap";
      }
    } else {
      open_buffered();
      mode_ = "no-mmap";
    }
    if (pin_requested_) {
      this->pin_memory();
    }
  }

  const std::uint8_t *data() const {
    return data_;
  }

  std::size_t size() const {
    return file_size_;
  }

  const std::string &mode() const {
    return mode_;
  }

  bool pin_applied() const {
    return pin_applied_;
  }

  const std::vector<std::string> &warnings() const {
    return warnings_;
  }

  void close() {
    if (data_ != nullptr && pin_applied_) {
#ifdef _WIN32
      VirtualUnlock(const_cast<std::uint8_t *>(data_), file_size_);
#else
      munlock(const_cast<std::uint8_t *>(data_), file_size_);
#endif
    }
#ifdef _WIN32
    if (data_ != nullptr && mapped_) {
      UnmapViewOfFile(data_);
    }
    if (mapping_handle_ != nullptr) {
      CloseHandle(mapping_handle_);
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(file_handle_);
    }
    mapping_handle_ = nullptr;
    file_handle_ = INVALID_HANDLE_VALUE;
#else
    if (data_ != nullptr && mapped_) {
      munmap(const_cast<std::uint8_t *>(data_), file_size_);
    }
    if (file_descriptor_ >= 0) {
      ::close(file_descriptor_);
    }
    file_descriptor_ = -1;
#endif
    buffer_.clear();
    buffer_.shrink_to_fit();
    data_ = nullptr;
    file_size_ = 0;
    mapped_ = false;
    pin_applied_ = false;
    pin_requested_ = false;
    mode_.clear();
    warnings_.clear();
  }

 private:
  std::filesystem::path file_path_;
  std::vector<std::uint8_t> buffer_;
  const std::uint8_t *data_ = nullptr;
  std::size_t file_size_ = 0;
  bool mapped_ = false;
  bool pin_requested_ = false;
  bool pin_applied_ = false;
  std::string mode_;
  std::vector<std::string> warnings_;

#ifdef _WIN32
  HANDLE file_handle_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_handle_ = nullptr;
#else
  int file_descriptor_ = -1;
#endif

  void open_buffered() {
    std::ifstream input(file_path_, std::ios::binary);
    if (!input) {
      throw std::runtime_error("Unable to open file for buffered reads.");
    }
    buffer_.resize(file_size_);
    if (file_size_ > 0) {
      input.read(reinterpret_cast<char *>(buffer_.data()), static_cast<std::streamsize>(file_size_));
      if (!input) {
        throw std::runtime_error("Failed to read the full GGUF file into memory.");
      }
    }
    data_ = buffer_.data();
    mapped_ = false;
  }

  // Ask the OS to stage the mapped GGUF into the page cache ahead of the
  // first-token walk, so cold-start tensor reads fault on warm pages instead
  // of trickling in 4 KB at a time. Advisory only: failures are ignored and
  // LIZARD_NATIVE_MMAP_PREFETCH=0 disables the hint entirely.
  void hint_sequential_prefetch() {
    if (data_ == nullptr || file_size_ == 0) {
      return;
    }
    static const bool enabled = [] {
      const char *value = std::getenv("LIZARD_NATIVE_MMAP_PREFETCH");
      return value == nullptr || std::string(value) != "0";
    }();
    if (!enabled) {
      return;
    }
#ifdef _WIN32
    // PrefetchVirtualMemory needs Windows 8+; resolve at runtime so the
    // binary keeps loading on any supported toolchain/header baseline.
    struct LizardMemoryRangeEntry {
      PVOID VirtualAddress;
      SIZE_T NumberOfBytes;
    };
    using PrefetchFn = BOOL(WINAPI *)(HANDLE, ULONG_PTR, LizardMemoryRangeEntry *, ULONG);
    static const PrefetchFn prefetch_fn = reinterpret_cast<PrefetchFn>(
      reinterpret_cast<void *>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory")));
    if (prefetch_fn != nullptr) {
      LizardMemoryRangeEntry range {};
      range.VirtualAddress = const_cast<std::uint8_t *>(data_);
      range.NumberOfBytes = file_size_;
      prefetch_fn(GetCurrentProcess(), 1, &range, 0);
    }
#else
    madvise(const_cast<std::uint8_t *>(data_), file_size_, MADV_WILLNEED);
#endif
  }

  void pin_memory() {
    if (data_ == nullptr || file_size_ == 0) {
      return;
    }
#ifdef _WIN32
    if (VirtualLock(const_cast<std::uint8_t *>(data_), file_size_) != 0) {
      pin_applied_ = true;
      return;
    }
    warnings_.push_back("VirtualLock failed; pinned memory was requested but could not be applied.");
#else
    if (mlock(const_cast<std::uint8_t *>(data_), file_size_) == 0) {
      pin_applied_ = true;
      return;
    }
    warnings_.push_back(std::string("mlock failed: ") + std::strerror(errno));
#endif
  }

  void open_mapped() {
#ifdef _WIN32
    file_handle_ = CreateFileW(
      file_path_.wstring().c_str(),
      GENERIC_READ,
      FILE_SHARE_READ,
      nullptr,
      OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL,
      nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("CreateFileW failed.");
    }
    mapping_handle_ = CreateFileMappingW(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_handle_ == nullptr) {
      throw std::runtime_error("CreateFileMappingW failed.");
    }
    data_ = static_cast<const std::uint8_t *>(MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
    if (data_ == nullptr) {
      throw std::runtime_error("MapViewOfFile failed.");
    }
    mapped_ = true;
#else
    file_descriptor_ = ::open(file_path_.c_str(), O_RDONLY);
    if (file_descriptor_ < 0) {
      throw std::runtime_error("open failed.");
    }
    void *mapped = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, file_descriptor_, 0);
    if (mapped == MAP_FAILED) {
      throw std::runtime_error("mmap failed.");
    }
    data_ = static_cast<const std::uint8_t *>(mapped);
    mapped_ = true;
#endif
  }
};

class ByteReader {
 public:
  ByteReader(const std::uint8_t *data, std::size_t size) : data_(data), size_(size), offset_(0) {}

  std::size_t offset() const {
    return offset_;
  }

  template <typename T>
  T read_primitive() {
    ensure(sizeof(T));
    T value {};
    std::memcpy(&value, data_ + offset_, sizeof(T));
    offset_ += sizeof(T);
    return value;
  }

  std::uint8_t read_u8() { return read_primitive<std::uint8_t>(); }
  std::int8_t read_i8() { return read_primitive<std::int8_t>(); }
  std::uint16_t read_u16() { return read_primitive<std::uint16_t>(); }
  std::int16_t read_i16() { return read_primitive<std::int16_t>(); }
  std::uint32_t read_u32() { return read_primitive<std::uint32_t>(); }
  std::int32_t read_i32() { return read_primitive<std::int32_t>(); }
  std::uint64_t read_u64() { return read_primitive<std::uint64_t>(); }
  std::int64_t read_i64() { return read_primitive<std::int64_t>(); }
  float read_f32() { return read_primitive<float>(); }
  double read_f64() { return read_primitive<double>(); }

  std::string read_string() {
    const std::uint64_t length = read_u64();
    ensure(static_cast<std::size_t>(length));
    std::string value(reinterpret_cast<const char *>(data_ + offset_), static_cast<std::size_t>(length));
    offset_ += static_cast<std::size_t>(length);
    return value;
  }

  void ensure(std::size_t bytes) const {
    if (offset_ + bytes > size_) {
      throw std::runtime_error("GGUF parser reached the end of the mapped file unexpectedly.");
    }
  }

 private:
  const std::uint8_t *data_;
  std::size_t size_;
  std::size_t offset_;
};

enum class GgufValueType : std::uint32_t {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12,
};

struct GgmlTypeLayout {
  const char *name;
  std::uint64_t block_size;
  std::uint64_t bytes_per_block;
};

static constexpr std::uint64_t k_qk_k = 256;
static constexpr std::uint64_t k_q4_k_scale_bytes = 12;

static std::optional<GgmlTypeLayout> ggml_type_layout(std::uint32_t ggml_type) {
  switch (ggml_type) {
    case 0:
      return GgmlTypeLayout {"F32", 1, 4};
    case 1:
      return GgmlTypeLayout {"F16", 1, 2};
    case 30:
      return GgmlTypeLayout {"BF16", 1, 2};
    case 2:
      return GgmlTypeLayout {"Q4_0", 32, 18};
    case 3:
      return GgmlTypeLayout {"Q4_1", 32, 20};
    case 6:
      return GgmlTypeLayout {"Q5_0", 32, 22};
    case 7:
      return GgmlTypeLayout {"Q5_1", 32, 24};
    case 8:
      return GgmlTypeLayout {"Q8_0", 32, 34};
    case 9:
      return GgmlTypeLayout {"Q8_1", 32, 36};
    case 10:
      return GgmlTypeLayout {"Q2_K", 256, 84};
    case 11:
      return GgmlTypeLayout {"Q3_K", 256, 110};
    case 12:
      return GgmlTypeLayout {"Q4_K", 256, 144};
    case 13:
      return GgmlTypeLayout {"Q5_K", 256, 176};
    case 14:
      return GgmlTypeLayout {"Q6_K", 256, 210};
    case 15:
      return GgmlTypeLayout {"Q8_K", 256, 292};
    case 23:
      return GgmlTypeLayout {"IQ4_XS", 256, 136};
    case 34:
      return GgmlTypeLayout {"TQ1_0", 256, 54};
    case 35:
      return GgmlTypeLayout {"TQ2_0", 256, 66};
    default:
      return std::nullopt;
  }
}

struct TensorInfo {
  std::string name;
  std::vector<std::uint64_t> dimensions;
  std::uint32_t ggml_type = 0;
  std::uint64_t offset = 0;
  std::string ggml_type_name;
  std::uint64_t element_count = 0;
  std::uint64_t byte_length = 0;
  std::uint64_t absolute_offset = 0;
};

struct ParsedGgufModel {
  std::filesystem::path file_path;
  std::string file_name;
  std::string memory_mode;
  bool pinned_memory = false;
  std::size_t file_size_bytes = 0;
  std::uint32_t version = 0;
  std::uint64_t tensor_count = 0;
  std::uint64_t metadata_kv_count = 0;
  std::uint64_t tensor_data_offset = 0;
  std::uint64_t tensor_data_bytes = 0;
  std::string metadata_hash;
  JsonObject metadata;
  std::vector<TensorInfo> tensors;
  std::string architecture;
  std::string quantization;
  std::string tokenizer;
  std::string chat_template;
  std::optional<std::uint64_t> context_length;
  std::optional<std::uint64_t> embedding_length;
  std::optional<std::uint64_t> block_count;
  std::optional<std::uint64_t> attention_head_count;
  std::optional<std::uint64_t> kv_head_count;
  std::optional<std::uint64_t> attention_key_length;
  std::optional<std::uint64_t> attention_value_length;
  std::optional<std::uint64_t> attention_key_length_swa;
  std::optional<std::uint64_t> attention_value_length_swa;
  std::optional<std::uint64_t> attention_sliding_window;
  std::optional<std::uint64_t> attention_shared_kv_layers;
  std::optional<std::uint64_t> embedding_length_per_layer_input;
  std::optional<double> rope_freq_base;
  std::optional<double> rope_freq_scale;
  std::optional<double> rope_freq_base_swa;
  std::optional<double> layer_norm_rms_epsilon;
  std::optional<double> final_logit_softcapping;
  std::optional<double> layer_norm_epsilon;
  std::optional<std::uint64_t> expert_count;
  std::optional<std::uint64_t> expert_used_count;
  std::optional<std::uint64_t> expert_shared_count;
  std::optional<double> expert_weights_scale;
  std::optional<std::uint64_t> expert_feed_forward_length;
  std::optional<std::uint64_t> expert_shared_feed_forward_length;
  std::optional<std::uint64_t> leading_dense_block_count;
  std::optional<std::uint64_t> rope_dimension_count;
  std::optional<double> rope_scaling_attn_factor;
  std::optional<std::uint64_t> rope_scaling_original_context_length;
  JsonObject special_tokens;
  JsonObject special_token_texts;
  std::vector<std::string> token_vocabulary;
  std::vector<std::string> token_merges;
  std::uint64_t token_vocabulary_size = 0;
  JsonArray token_preview;
  std::uint64_t token_merge_count = 0;
  JsonArray merge_preview;
  JsonObject executor_shape;
  std::optional<double> parameter_estimate_b;
  std::vector<std::string> warnings;
};

static constexpr std::size_t k_missing_tensor_index = std::numeric_limits<std::size_t>::max();

struct LlamaLayerTensorIndex {
  std::size_t attention_norm = k_missing_tensor_index;
  std::size_t query = k_missing_tensor_index;
  std::size_t key = k_missing_tensor_index;
  std::size_t value = k_missing_tensor_index;
  std::size_t output = k_missing_tensor_index;
  std::size_t query_norm = k_missing_tensor_index;
  std::size_t key_norm = k_missing_tensor_index;
  std::size_t ffn_norm = k_missing_tensor_index;
  std::size_t gate = k_missing_tensor_index;
  std::size_t down = k_missing_tensor_index;
  std::size_t up = k_missing_tensor_index;
  std::size_t post_attention_norm = k_missing_tensor_index;
  std::size_t post_ffw_norm = k_missing_tensor_index;
  std::size_t post_norm = k_missing_tensor_index;
  std::size_t per_layer_input_gate = k_missing_tensor_index;
  std::size_t per_layer_projection = k_missing_tensor_index;
  std::size_t layer_output_scale = k_missing_tensor_index;
  // GGUF family expansion (qwen2/qwen3/phi/phi3/glm4/starcoder2/MoE groups).
  std::size_t qkv_fused = k_missing_tensor_index;
  std::size_t qkv_fused_bias = k_missing_tensor_index;
  std::size_t attention_norm_bias = k_missing_tensor_index;
  std::size_t query_bias = k_missing_tensor_index;
  std::size_t key_bias = k_missing_tensor_index;
  std::size_t value_bias = k_missing_tensor_index;
  std::size_t output_bias = k_missing_tensor_index;
  std::size_t ffn_norm_bias = k_missing_tensor_index;
  std::size_t up_bias = k_missing_tensor_index;
  std::size_t down_bias = k_missing_tensor_index;
  std::size_t moe_gate_inp = k_missing_tensor_index;
  std::size_t moe_gate_exps = k_missing_tensor_index;
  std::size_t moe_up_exps = k_missing_tensor_index;
  std::size_t moe_down_exps = k_missing_tensor_index;
  std::size_t shexp_gate_inp = k_missing_tensor_index;
  std::size_t shexp_gate = k_missing_tensor_index;
  std::size_t shexp_up = k_missing_tensor_index;
  std::size_t shexp_down = k_missing_tensor_index;
};

struct LlamaTensorIndex {
  std::size_t token_embeddings = k_missing_tensor_index;
  std::size_t output_norm = k_missing_tensor_index;
  std::size_t output = k_missing_tensor_index;
  std::size_t output_norm_bias = k_missing_tensor_index;
  std::size_t output_bias = k_missing_tensor_index;
  std::size_t rope_factors_long = k_missing_tensor_index;
  std::size_t rope_factors_short = k_missing_tensor_index;
  std::size_t rope_freqs = k_missing_tensor_index;
  std::size_t per_layer_token_embeddings = k_missing_tensor_index;
  std::size_t per_layer_model_projection = k_missing_tensor_index;
  std::size_t per_layer_projection_norm = k_missing_tensor_index;
  bool output_tied = false;
  std::vector<LlamaLayerTensorIndex> layers;
  std::vector<std::string> missing_shared;
  JsonArray missing_layers;
  JsonArray supported_dequant_types;
  bool ready = false;
};

static std::string fnv1a_64_hex(const std::uint8_t *data, std::size_t size) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::size_t index = 0; index < size; index += 1) {
    hash ^= static_cast<std::uint64_t>(data[index]);
    hash *= 1099511628211ULL;
  }
  std::ostringstream output;
  output << std::hex << std::setw(16) << std::setfill('0') << hash;
  return output.str();
}

static std::optional<std::uint64_t> json_to_u64(const JsonValue *value) {
  if (value != nullptr && value->is_number()) {
    return static_cast<std::uint64_t>(value->as_number());
  }
  return std::nullopt;
}

static std::optional<double> json_to_double(const JsonValue *value) {
  if (value != nullptr && value->is_number()) {
    return value->as_number();
  }
  return std::nullopt;
}

static std::string json_to_string(const JsonValue *value, const std::string &fallback = "") {
  return value != nullptr && value->is_string() ? value->as_string() : fallback;
}

static JsonValue read_metadata_value(ByteReader &reader, GgufValueType type) {
  switch (type) {
    case GgufValueType::Uint8:
      return JsonValue(static_cast<std::uint64_t>(reader.read_u8()));
    case GgufValueType::Int8:
      return JsonValue(static_cast<double>(reader.read_i8()));
    case GgufValueType::Uint16:
      return JsonValue(static_cast<std::uint64_t>(reader.read_u16()));
    case GgufValueType::Int16:
      return JsonValue(static_cast<double>(reader.read_i16()));
    case GgufValueType::Uint32:
      return JsonValue(static_cast<std::uint64_t>(reader.read_u32()));
    case GgufValueType::Int32:
      return JsonValue(static_cast<double>(reader.read_i32()));
    case GgufValueType::Float32:
      return JsonValue(static_cast<double>(reader.read_f32()));
    case GgufValueType::Bool:
      return JsonValue(reader.read_u8() != 0);
    case GgufValueType::String:
      return JsonValue(reader.read_string());
    case GgufValueType::Array: {
      const auto element_type = static_cast<GgufValueType>(reader.read_u32());
      const std::uint64_t length = reader.read_u64();
      JsonArray values;
      values.reserve(static_cast<std::size_t>(length));
      for (std::uint64_t index = 0; index < length; index += 1) {
        values.push_back(read_metadata_value(reader, element_type));
      }
      return JsonValue(values);
    }
    case GgufValueType::Uint64:
      return JsonValue(reader.read_u64());
    case GgufValueType::Int64:
      return JsonValue(static_cast<double>(reader.read_i64()));
    case GgufValueType::Float64:
      return JsonValue(reader.read_f64());
    default:
      throw std::runtime_error("Unsupported GGUF metadata value type.");
  }
}

static std::string infer_quantization_from_file_type(std::uint64_t value) {
  switch (value) {
    case 0:
      return "f32";
    case 1:
      return "f16";
    case 2:
      return "q4_0";
    case 3:
      return "q4_1";
    case 7:
      return "q8_0";
    case 8:
      return "q5_0";
    case 9:
      return "q5_1";
    case 10:
      return "q2_k";
    case 11:
      return "q3_k_s";
    case 12:
      return "q3_k_m";
    case 13:
      return "q3_k_l";
    case 14:
      return "q4_k_s";
    case 15:
      return "q4_k_m";
    case 16:
      return "q5_k_s";
    case 17:
      return "q5_k_m";
    case 18:
      return "q6_k";
    case 19:
      return "iq2_xxs";
    case 20:
      return "iq2_xs";
    case 21:
      return "q2_k_s";
    case 22:
      return "iq3_xs";
    case 23:
      return "iq3_xxs";
    case 24:
      return "iq1_s";
    case 25:
      return "iq4_nl";
    case 26:
      return "iq3_s";
    case 27:
      return "iq3_m";
    case 28:
      return "iq2_s";
    case 29:
      return "iq2_m";
    case 30:
      return "iq4_xs";
    case 31:
      return "iq1_m";
    case 32:
      return "bf16";
    case 36:
      return "tq1_0";
    case 37:
      return "tq2_0";
    default:
      return "";
  }
}

static std::string infer_quantization_from_filename(const std::string &file_name) {
  const std::string upper_name = to_upper(file_name);
  const std::vector<std::pair<std::string, std::string>> patterns = {
    {"Q4_K_M", "q4_k_m"},
    {"IQ4_XS", "iq4_xs"},
    {"TQ1_0", "tq1_0"},
    {"TQ2_0", "tq2_0"},
    {"Q8_0", "q8_0"},
    {"Q6_K", "q6_k"},
    {"Q5_K_M", "q5_k_m"},
    {"Q3_K_M", "q3_k_m"},
    {"Q2_K", "q2_k"},
    {"Q4_0", "q4_0"},
    {"F16", "f16"},
    {"F32", "f32"},
    {"BF16", "bf16"},
  };
  for (const auto &[pattern, normalized] : patterns) {
    if (upper_name.find(pattern) != std::string::npos) {
      return normalized;
    }
  }
  return "";
}

static const JsonValue *lookup_first(const JsonObject &metadata, const std::vector<std::string> &keys, const std::vector<std::string> &suffixes = {}) {
  for (const std::string &key : keys) {
    if (const JsonValue *value = find_object_value(metadata, key)) {
      return value;
    }
  }
  for (const auto &[key, value] : metadata) {
    for (const std::string &suffix : suffixes) {
      if (ends_with_ci(key, suffix)) {
        return &value;
      }
    }
  }
  return nullptr;
}

static const TensorInfo *find_tensor_by_name(const std::vector<TensorInfo> &tensors, const std::vector<std::string> &names) {
  for (const std::string &name : names) {
    for (const TensorInfo &tensor : tensors) {
      if (tensor.name == name) {
        return &tensor;
      }
    }
  }
  return nullptr;
}

static std::size_t find_tensor_index_by_name(const std::vector<TensorInfo> &tensors, const std::vector<std::string> &names) {
  for (const std::string &name : names) {
    for (std::size_t index = 0; index < tensors.size(); index += 1) {
      if (tensors[index].name == name) {
        return index;
      }
    }
  }
  return k_missing_tensor_index;
}

static bool is_gemma_architecture(const std::string &architecture) {
  const std::string normalized = to_lower(architecture);
  return normalized == "gemma" || normalized == "gemma2" || normalized == "gemma3" || normalized == "gemma4";
}

// --- GGUF family expansion -------------------------------------------------
// One parameterized execution path covers the single-wave family groups
// (qwen2, qwen3, mistral, mixtral, phi, phi3/phi4, qwen2moe, qwen3moe,
// deepseek, glm, starcoder2). The llama and gemma paths stay untouched;
// models that resolve a FamilySpec are excluded from the llama path so a
// qwen2-style GGUF can never silently run through the bias-free llama graph.

enum class FamilyNormKind { rms, layer };
enum class FamilyRopeKind { norm, neox };
enum class FamilyFfnKind { swiglu_split, swiglu_fused, gelu_seq };

struct FamilySpec {
  std::string family;                // canonical group id reported on all surfaces
  FamilyNormKind norm_kind = FamilyNormKind::rms;
  FamilyRopeKind rope_kind = FamilyRopeKind::neox;
  bool rope_partial = false;         // rotate only rope_dimension_count dims per head
  bool rope_long_short_factors = false; // phi3 longrope factor tensors
  bool qk_norm = false;              // per-head RMS norm on q/k (qwen3 lineage)
  FamilyFfnKind ffn_kind = FamilyFfnKind::swiglu_split;
  bool parallel_residual = false;    // phi2: attention and FFN share the pre-norm input
  bool has_ffn_norm = true;          // phi2 has no separate FFN norm
  bool post_attention_norm = false;  // glm4: norm applied to attn output before residual
  bool post_ffw_norm = false;        // glm4: norm applied to FFN output before residual
  bool moe = false;
  bool moe_norm_topk_weights = false;
  bool moe_shared_expert = false;
  bool moe_shared_expert_gated = false; // qwen2moe sigmoid-gated shared expert
};

static std::optional<FamilySpec> resolve_family_spec(const ParsedGgufModel &model) {
  const std::string arch = to_lower(model.architecture);
  FamilySpec spec;
  if (arch == "qwen2") {
    spec.family = "qwen2";
    return spec;
  }
  if (arch == "qwen3") {
    spec.family = "qwen3";
    spec.qk_norm = true;
    return spec;
  }
  if (arch == "mistral") {
    // Mistral GGUFs normally convert as `llama`; accept the literal arch too.
    spec.family = "mistral";
    spec.rope_kind = FamilyRopeKind::norm;
    return spec;
  }
  if (arch == "llama" && model.expert_count.value_or(0) > 0) {
    spec.family = "mixtral";
    spec.rope_kind = FamilyRopeKind::norm;
    spec.moe = true;
    spec.moe_norm_topk_weights = true;
    return spec;
  }
  if (arch == "phi3") {
    // phi3 covers Phi-3, Phi-3.5, and Phi-4 GGUF conversions.
    spec.family = "phi3";
    spec.rope_partial = true;
    spec.rope_long_short_factors = true;
    spec.ffn_kind = FamilyFfnKind::swiglu_fused;
    return spec;
  }
  if (arch == "phi2") {
    spec.family = "phi";
    spec.norm_kind = FamilyNormKind::layer;
    spec.rope_partial = true;
    spec.ffn_kind = FamilyFfnKind::gelu_seq;
    spec.parallel_residual = true;
    spec.has_ffn_norm = false;
    return spec;
  }
  if (arch == "glm4") {
    spec.family = "glm";
    spec.rope_kind = FamilyRopeKind::norm;
    spec.rope_partial = true;
    spec.ffn_kind = FamilyFfnKind::swiglu_fused;
    spec.post_attention_norm = true;
    spec.post_ffw_norm = true;
    return spec;
  }
  if (arch == "starcoder2") {
    spec.family = "starcoder2";
    spec.norm_kind = FamilyNormKind::layer;
    spec.ffn_kind = FamilyFfnKind::gelu_seq;
    return spec;
  }
  if (arch == "qwen2moe") {
    spec.family = "qwen2moe";
    spec.moe = true;
    spec.moe_shared_expert = true;
    spec.moe_shared_expert_gated = true;
    return spec;
  }
  if (arch == "qwen3moe") {
    spec.family = "qwen3moe";
    spec.qk_norm = true;
    spec.moe = true;
    spec.moe_norm_topk_weights = true;
    return spec;
  }
  if (arch == "deepseek") {
    spec.family = "deepseek";
    spec.rope_kind = FamilyRopeKind::norm;
    spec.moe = true;
    spec.moe_shared_expert = true;
    return spec;
  }
  return std::nullopt;
}

// Target-list architectures that are truly not executable yet. Every entry
// needs the exact missing worker capability; these surface as `blocked` with
// a `will be updated soon` marker instead of silently switching providers.
static const char *family_blocked_reason(const std::string &architecture) {
  const std::string arch = to_lower(architecture);
  if (arch == "deepseek2" || arch == "deepseek3") {
    return "deepseek2/deepseek3 GGUFs require Multi-head Latent Attention "
      "(q_a/kv_a latent compression projections and decoupled RoPE keys), which the "
      "native worker does not implement yet. Status: blocked; will be updated soon.";
  }
  if (arch == "chatglm") {
    return "Legacy chatglm GGUFs use the pre-GLM4 packed QKV/rope layout, which the "
      "native worker does not implement yet; the glm4 lineage is supported natively. "
      "Status: blocked; will be updated soon.";
  }
  return nullptr;
}

// Decodes the two byte-level BPE markers that matter for stop matching back to
// real characters: GPT-2 byte BPE maps space -> U+0120 ("Ġ", UTF-8 C4 A0) and
// newline -> U+010A ("Ċ", UTF-8 C4 8A). Raw vocab token text carries these, so
// a text stop string like "### " or "\nUser:" never matches the raw stream
// without this. Printable ASCII (# < > | : letters) maps to itself, so those
// need no decoding.
static std::string decode_bpe_whitespace(const std::string &text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t index = 0; index < text.size(); index += 1) {
    if (index + 1 < text.size() && static_cast<unsigned char>(text[index]) == 0xC4) {
      const unsigned char next = static_cast<unsigned char>(text[index + 1]);
      if (next == 0xA0) { out.push_back(' '); index += 1; continue; }   // Ġ
      if (next == 0x8A) { out.push_back('\n'); index += 1; continue; }  // Ċ
    }
    out.push_back(text[index]);
  }
  return out;
}

static std::vector<std::string> json_array_to_strings(const JsonValue *value) {
  std::vector<std::string> items;
  if (value == nullptr || !value->is_array()) {
    return items;
  }
  for (const JsonValue &entry : value->as_array()) {
    if (entry.is_string()) {
      items.push_back(entry.as_string());
    }
  }
  return items;
}

static std::vector<std::uint64_t> json_array_to_u64s(const JsonValue *value) {
  std::vector<std::uint64_t> items;
  if (value == nullptr || !value->is_array()) {
    return items;
  }
  for (const JsonValue &entry : value->as_array()) {
    if (entry.is_number()) {
      items.push_back(static_cast<std::uint64_t>(entry.as_number()));
    }
  }
  return items;
}

static std::uint64_t align_offset(std::uint64_t offset, std::uint64_t alignment) {
  const std::uint64_t normalized = alignment == 0 ? 32 : alignment;
  const std::uint64_t remainder = offset % normalized;
  return remainder == 0 ? offset : offset + normalized - remainder;
}

static std::uint64_t tensor_element_count(const std::vector<std::uint64_t> &dimensions) {
  std::uint64_t total = 1;
  for (std::uint64_t dimension : dimensions) {
    total *= dimension;
  }
  return total;
}

static std::vector<std::size_t> sort_token_indices_by_match_priority(const std::vector<std::string> &tokens) {
  std::vector<std::size_t> indices(tokens.size());
  for (std::size_t index = 0; index < tokens.size(); index += 1) {
    indices[index] = index;
  }
  std::sort(indices.begin(), indices.end(), [&](std::size_t left, std::size_t right) {
    if (tokens[left].size() != tokens[right].size()) {
      return tokens[left].size() > tokens[right].size();
    }
    return tokens[left] < tokens[right];
  });
  return indices;
}

static float half_to_float(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16;
  const std::uint32_t exponent = (value >> 10) & 0x1FU;
  const std::uint32_t mantissa = value & 0x03FFU;

  std::uint32_t f32_bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      f32_bits = sign;
    } else {
      std::uint32_t normalized_mantissa = mantissa;
      std::int32_t normalized_exponent = -14;
      while ((normalized_mantissa & 0x0400U) == 0) {
        normalized_mantissa <<= 1;
        normalized_exponent -= 1;
      }
      normalized_mantissa &= 0x03FFU;
      f32_bits = sign
        | static_cast<std::uint32_t>((normalized_exponent + 127) << 23)
        | (normalized_mantissa << 13);
    }
  } else if (exponent == 0x1FU) {
    f32_bits = sign | 0x7F800000U | (mantissa << 13);
  } else {
    f32_bits = sign
      | static_cast<std::uint32_t>(((static_cast<std::int32_t>(exponent) - 15) + 127) << 23)
      | (mantissa << 13);
  }

  float output = 0.0f;
  std::memcpy(&output, &f32_bits, sizeof(output));
  return output;
}

static float bfloat16_to_float(std::uint16_t value) {
  const std::uint32_t f32_bits = static_cast<std::uint32_t>(value) << 16;
  float output = 0.0f;
  std::memcpy(&output, &f32_bits, sizeof(output));
  return output;
}

static std::uint16_t read_u16_le(const std::uint8_t *data) {
  std::uint16_t value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

static float read_f32_le(const std::uint8_t *data) {
  float value = 0.0f;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

static std::string utf8_from_codepoint(std::uint32_t codepoint) {
  std::string output;
  if (codepoint <= 0x7F) {
    output.push_back(static_cast<char>(codepoint));
    return output;
  }
  if (codepoint <= 0x7FF) {
    output.push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    return output;
  }
  if (codepoint <= 0xFFFF) {
    output.push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    return output;
  }
  output.push_back(static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07)));
  output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
  output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
  output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  return output;
}

static const std::vector<std::string> &gpt2_byte_to_unicode_symbols() {
  static const std::vector<std::string> symbols = []() {
    std::vector<int> bytes;
    std::vector<std::uint32_t> chars;
    for (int code = 33; code <= 126; code += 1) {
      bytes.push_back(code);
      chars.push_back(static_cast<std::uint32_t>(code));
    }
    for (int code = 161; code <= 172; code += 1) {
      bytes.push_back(code);
      chars.push_back(static_cast<std::uint32_t>(code));
    }
    for (int code = 174; code <= 255; code += 1) {
      bytes.push_back(code);
      chars.push_back(static_cast<std::uint32_t>(code));
    }
    std::set<int> seen(bytes.begin(), bytes.end());
    int extra = 0;
    for (int byte = 0; byte < 256; byte += 1) {
      if (seen.find(byte) != seen.end()) {
        continue;
      }
      bytes.push_back(byte);
      chars.push_back(static_cast<std::uint32_t>(256 + extra));
      extra += 1;
    }
    std::vector<std::string> output(256);
    for (std::size_t index = 0; index < bytes.size(); index += 1) {
      output[static_cast<std::size_t>(bytes[index])] = utf8_from_codepoint(chars[index]);
    }
    return output;
  }();
  return symbols;
}

struct Utf8Unit {
  std::string bytes;
  std::uint32_t codepoint = 0;
};

static std::vector<Utf8Unit> split_utf8_units(const std::string &text) {
  std::vector<Utf8Unit> units;
  std::size_t offset = 0;
  while (offset < text.size()) {
    const unsigned char lead = static_cast<unsigned char>(text[offset]);
    std::size_t length = 1;
    std::uint32_t codepoint = lead;
    if ((lead & 0xE0) == 0xC0 && offset + 1 < text.size()) {
      length = 2;
      codepoint = static_cast<std::uint32_t>(lead & 0x1F) << 6;
      codepoint |= static_cast<std::uint32_t>(static_cast<unsigned char>(text[offset + 1]) & 0x3F);
    } else if ((lead & 0xF0) == 0xE0 && offset + 2 < text.size()) {
      length = 3;
      codepoint = static_cast<std::uint32_t>(lead & 0x0F) << 12;
      codepoint |= static_cast<std::uint32_t>(static_cast<unsigned char>(text[offset + 1]) & 0x3F) << 6;
      codepoint |= static_cast<std::uint32_t>(static_cast<unsigned char>(text[offset + 2]) & 0x3F);
    } else if ((lead & 0xF8) == 0xF0 && offset + 3 < text.size()) {
      length = 4;
      codepoint = static_cast<std::uint32_t>(lead & 0x07) << 18;
      codepoint |= static_cast<std::uint32_t>(static_cast<unsigned char>(text[offset + 1]) & 0x3F) << 12;
      codepoint |= static_cast<std::uint32_t>(static_cast<unsigned char>(text[offset + 2]) & 0x3F) << 6;
      codepoint |= static_cast<std::uint32_t>(static_cast<unsigned char>(text[offset + 3]) & 0x3F);
    }
    units.push_back(Utf8Unit {text.substr(offset, length), codepoint});
    offset += length;
  }
  return units;
}

static bool utf8_unit_is_space(const Utf8Unit &unit) {
  return unit.bytes.size() == 1 && std::isspace(static_cast<unsigned char>(unit.bytes[0])) != 0;
}

static bool utf8_unit_is_digit(const Utf8Unit &unit) {
  return unit.bytes.size() == 1 && std::isdigit(static_cast<unsigned char>(unit.bytes[0])) != 0;
}

static bool utf8_unit_is_letterish(const Utf8Unit &unit) {
  return unit.bytes.size() == 1
    ? std::isalpha(static_cast<unsigned char>(unit.bytes[0])) != 0
    : true;
}

static std::optional<std::vector<std::uint64_t>> tokenize_exact_vocab(
  const std::string &text,
  const std::vector<std::string> &token_vocabulary,
  const std::vector<std::size_t> &sorted_token_indices) {
  if (token_vocabulary.empty() || sorted_token_indices.empty()) {
    return std::nullopt;
  }

  std::vector<std::uint64_t> token_ids;
  std::size_t offset = 0;
  while (offset < text.size()) {
    bool matched = false;
    for (std::size_t token_index : sorted_token_indices) {
      const std::string &token = token_vocabulary[token_index];
      if (token.empty()) {
        continue;
      }
      if (text.compare(offset, token.size(), token) == 0) {
        token_ids.push_back(static_cast<std::uint64_t>(token_index));
        offset += token.size();
        matched = true;
        break;
      }
    }
    if (!matched) {
      return std::nullopt;
    }
  }
  return token_ids;
}

static void prepend_bos_if_missing(
  std::vector<std::uint64_t> &token_ids,
  const ParsedGgufModel &model) {
  const auto bos_token = json_to_u64(find_object_value(model.special_tokens, "bos"));
  if (!bos_token.has_value() || token_ids.empty() || token_ids.front() == bos_token.value()) {
    return;
  }
  token_ids.insert(token_ids.begin(), bos_token.value());
}

static std::vector<std::string> split_gpt2_bpe_pieces_utf8(const std::string &text) {
  std::vector<std::string> pieces;
  const std::vector<Utf8Unit> units = split_utf8_units(text);
  std::size_t offset = 0;
  auto join_units = [&](std::size_t start, std::size_t end) {
    std::string joined;
    for (std::size_t index = start; index < end; index += 1) {
      joined += units[index].bytes;
    }
    return joined;
  };

  while (offset < units.size()) {
    const Utf8Unit &current = units[offset];
    if (current.bytes == "'") {
      const std::vector<std::string> contractions = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
      bool matched = false;
      for (const std::string &contraction : contractions) {
        std::string candidate = join_units(offset, std::min(units.size(), offset + contraction.size()));
        if (candidate == contraction) {
          pieces.push_back(contraction);
          offset += contraction.size();
          matched = true;
          break;
        }
      }
      if (matched) {
        continue;
      }
    }

    if (utf8_unit_is_space(current)) {
      const std::size_t run_start = offset;
      while (offset < units.size() && utf8_unit_is_space(units[offset])) {
        offset += 1;
      }
      if (offset >= units.size()) {
        pieces.push_back(join_units(run_start, offset));
        break;
      }
      const Utf8Unit &next = units[offset];
      if (utf8_unit_is_letterish(next)) {
        const std::size_t token_start = run_start;
        while (offset < units.size() && utf8_unit_is_letterish(units[offset])) {
          offset += 1;
        }
        pieces.push_back(join_units(token_start, offset));
        continue;
      }
      if (utf8_unit_is_digit(next)) {
        const std::size_t token_start = run_start;
        while (offset < units.size() && utf8_unit_is_digit(units[offset])) {
          offset += 1;
        }
        pieces.push_back(join_units(token_start, offset));
        continue;
      }
      if (!utf8_unit_is_space(next)
        && !utf8_unit_is_letterish(next)
        && !utf8_unit_is_digit(next)) {
        const std::size_t token_start = run_start;
        while (offset < units.size()
          && !utf8_unit_is_space(units[offset])
          && !utf8_unit_is_letterish(units[offset])
          && !utf8_unit_is_digit(units[offset])) {
          offset += 1;
        }
        pieces.push_back(join_units(token_start, offset));
        continue;
      }
      pieces.push_back(join_units(run_start, offset));
      continue;
    }

    if (utf8_unit_is_letterish(current)) {
      const std::size_t start = offset;
      while (offset < units.size() && utf8_unit_is_letterish(units[offset])) {
        offset += 1;
      }
      pieces.push_back(join_units(start, offset));
      continue;
    }
    if (utf8_unit_is_digit(current)) {
      const std::size_t start = offset;
      while (offset < units.size() && utf8_unit_is_digit(units[offset])) {
        offset += 1;
      }
      pieces.push_back(join_units(start, offset));
      continue;
    }
    if (utf8_unit_is_space(current)) {
      const std::size_t start = offset;
      while (offset < units.size() && utf8_unit_is_space(units[offset])) {
        offset += 1;
      }
      pieces.push_back(join_units(start, offset));
      continue;
    }
    const std::size_t start = offset;
    while (offset < units.size()
      && !utf8_unit_is_space(units[offset])
      && !utf8_unit_is_letterish(units[offset])
      && !utf8_unit_is_digit(units[offset])) {
      offset += 1;
    }
    pieces.push_back(join_units(start, offset));
  }
  return pieces;
}

static std::vector<std::string> encode_piece_to_byte_bpe_symbols(const std::string &piece) {
  const std::vector<std::string> &encoder = gpt2_byte_to_unicode_symbols();
  std::vector<std::string> output;
  output.reserve(piece.size());
  for (unsigned char byte : piece) {
    output.push_back(encoder[static_cast<std::size_t>(byte)]);
  }
  return output;
}

static std::optional<std::vector<std::uint64_t>> tokenize_gpt2_bpe_ascii(
  const std::string &text,
  const ParsedGgufModel &model,
  const std::map<std::string, std::size_t> &token_to_id,
  const std::map<std::string, std::size_t> &merge_ranks) {
  if (token_to_id.empty() || merge_ranks.empty()) {
    return std::nullopt;
  }
  std::vector<std::uint64_t> token_ids;
  for (const std::string &piece : split_gpt2_bpe_pieces_utf8(text)) {
    std::vector<std::string> symbols = encode_piece_to_byte_bpe_symbols(piece);
    if (symbols.empty()) {
      continue;
    }
    while (symbols.size() > 1) {
      std::size_t best_index = std::numeric_limits<std::size_t>::max();
      std::size_t best_rank = std::numeric_limits<std::size_t>::max();
      for (std::size_t index = 0; index + 1 < symbols.size(); index += 1) {
        const std::string pair = symbols[index] + " " + symbols[index + 1];
        auto iterator = merge_ranks.find(pair);
        if (iterator == merge_ranks.end() || iterator->second >= best_rank) {
          continue;
        }
        best_rank = iterator->second;
        best_index = index;
      }
      if (best_index == std::numeric_limits<std::size_t>::max()) {
        break;
      }
      symbols[best_index] += symbols[best_index + 1];
      symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }
    for (const std::string &symbol : symbols) {
      auto iterator = token_to_id.find(symbol);
      if (iterator == token_to_id.end()) {
        return std::nullopt;
      }
      token_ids.push_back(static_cast<std::uint64_t>(iterator->second));
    }
  }
  prepend_bos_if_missing(token_ids, model);
  return token_ids;
}

static JsonObject analyze_llama_executor_shape(const ParsedGgufModel &model) {
  std::vector<std::string> tensor_type_names;
  for (const TensorInfo &tensor : model.tensors) {
    const std::string type_name = tensor.ggml_type_name.empty()
      ? "TYPE_" + std::to_string(tensor.ggml_type)
      : tensor.ggml_type_name;
    if (std::find(tensor_type_names.begin(), tensor_type_names.end(), type_name) == tensor_type_names.end()) {
      tensor_type_names.push_back(type_name);
    }
  }
  JsonArray tensor_types;
  for (const std::string &type_name : tensor_type_names) {
    tensor_types.push_back(JsonValue(type_name));
  }
  const bool has_token_embeddings = find_tensor_by_name(model.tensors, {"token_embd.weight", "tok_embeddings.weight"}) != nullptr;
  const bool has_output = find_tensor_by_name(model.tensors, {"output.weight", "lm_head.weight"}) != nullptr;
  const bool output_tied = !has_output && has_token_embeddings;
  JsonObject shared = {
    {"tokenEmbeddings", JsonValue(has_token_embeddings)},
    {"outputNorm", JsonValue(find_tensor_by_name(model.tensors, {"output_norm.weight", "norm.weight"}) != nullptr)},
    {"output", JsonValue(has_output || output_tied)},
    {"outputTied", JsonValue(output_tied)},
  };
  const std::uint64_t total_layers = model.block_count.value_or(0);
  std::uint64_t complete_layers = 0;
  JsonArray incomplete_layers;

  for (std::uint64_t layer = 0; layer < total_layers; layer += 1) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const std::vector<std::pair<std::string, bool>> requirements = {
      {"attentionNorm", find_tensor_by_name(model.tensors, {prefix + "attn_norm.weight"}) != nullptr},
      {"query", find_tensor_by_name(model.tensors, {prefix + "attn_q.weight"}) != nullptr},
      {"key", find_tensor_by_name(model.tensors, {prefix + "attn_k.weight"}) != nullptr},
      {"value", find_tensor_by_name(model.tensors, {prefix + "attn_v.weight"}) != nullptr},
      {"output", find_tensor_by_name(model.tensors, {prefix + "attn_output.weight"}) != nullptr},
      {"ffnNorm", find_tensor_by_name(model.tensors, {prefix + "ffn_norm.weight"}) != nullptr},
      {"gate", find_tensor_by_name(model.tensors, {prefix + "ffn_gate.weight"}) != nullptr},
      {"down", find_tensor_by_name(model.tensors, {prefix + "ffn_down.weight"}) != nullptr},
      {"up", find_tensor_by_name(model.tensors, {prefix + "ffn_up.weight"}) != nullptr},
    };
    JsonArray missing;
    for (const auto &[name, present] : requirements) {
      if (!present) {
        missing.push_back(JsonValue(name));
      }
    }
    if (missing.empty()) {
      complete_layers += 1;
      continue;
    }
    incomplete_layers.push_back(JsonValue(JsonObject {
      {"layer", JsonValue(layer)},
      {"missing", JsonValue(missing)},
    }));
  }

  const bool shared_ready = json_bool_or_default(shared, "tokenEmbeddings", false)
    && json_bool_or_default(shared, "outputNorm", false)
    && json_bool_or_default(shared, "output", false);

  return {
    {"shared", JsonValue(shared)},
    {"tensorTypes", JsonValue(tensor_types)},
    {"totalLayers", JsonValue(total_layers)},
    {"completeLayers", JsonValue(complete_layers)},
    {"incompleteLayers", JsonValue(incomplete_layers)},
    {"tensorDirectoryComplete", JsonValue(shared_ready && (total_layers == 0 || complete_layers == total_layers))},
  };
}

// Per-layer required-tensor labels for a FamilySpec model. Used both by the
// executor-shape analyzer (name-based reporting) and the tensor index missing
// computation so every surface reports the same gaps.
static std::vector<std::string> family_layer_missing_labels(
  const ParsedGgufModel &model,
  const FamilySpec &spec,
  std::uint64_t layer) {
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  const auto has = [&](const std::string &name) {
    return find_tensor_by_name(model.tensors, {prefix + name}) != nullptr;
  };
  std::vector<std::string> missing;
  if (!has("attn_norm.weight")) missing.push_back("attentionNorm");
  const bool fused_qkv = has("attn_qkv.weight");
  if (!fused_qkv) {
    if (!has("attn_q.weight")) missing.push_back("query");
    if (!has("attn_k.weight")) missing.push_back("key");
    if (!has("attn_v.weight")) missing.push_back("value");
  }
  if (!has("attn_output.weight")) missing.push_back("output");
  if (spec.qk_norm) {
    if (!has("attn_q_norm.weight")) missing.push_back("queryNorm");
    if (!has("attn_k_norm.weight")) missing.push_back("keyNorm");
  }
  if (spec.post_attention_norm && !has("post_attention_norm.weight")) missing.push_back("postAttentionNorm");
  if (spec.post_ffw_norm && !has("post_ffw_norm.weight")) missing.push_back("postFfwNorm");
  if (spec.has_ffn_norm && !has("ffn_norm.weight")) missing.push_back("ffnNorm");

  // deepseek keeps its leading blocks dense; detect dense-vs-expert per layer
  // by tensor presence so the report never guesses.
  const bool dense_layer = !spec.moe
    || (has("ffn_gate.weight") && !has("ffn_gate_exps.weight"));
  if (!dense_layer) {
    if (!has("ffn_gate_inp.weight")) missing.push_back("moeGateInp");
    if (!has("ffn_gate_exps.weight")) missing.push_back("moeGateExps");
    if (!has("ffn_up_exps.weight")) missing.push_back("moeUpExps");
    if (!has("ffn_down_exps.weight")) missing.push_back("moeDownExps");
    if (spec.moe_shared_expert) {
      if (!has("ffn_gate_shexp.weight")) missing.push_back("sharedExpertGate");
      if (!has("ffn_up_shexp.weight")) missing.push_back("sharedExpertUp");
      if (!has("ffn_down_shexp.weight")) missing.push_back("sharedExpertDown");
      if (spec.moe_shared_expert_gated && !has("ffn_gate_inp_shexp.weight")) missing.push_back("sharedExpertGateInp");
    }
  } else {
    if (spec.ffn_kind == FamilyFfnKind::swiglu_split && !has("ffn_gate.weight")) missing.push_back("gate");
    if (!has("ffn_down.weight")) missing.push_back("down");
    if (!has("ffn_up.weight")) missing.push_back("up");
  }
  return missing;
}

static JsonObject analyze_family_executor_shape(const ParsedGgufModel &model, const FamilySpec &spec) {
  std::vector<std::string> tensor_type_names;
  for (const TensorInfo &tensor : model.tensors) {
    const std::string type_name = tensor.ggml_type_name.empty()
      ? "TYPE_" + std::to_string(tensor.ggml_type)
      : tensor.ggml_type_name;
    if (std::find(tensor_type_names.begin(), tensor_type_names.end(), type_name) == tensor_type_names.end()) {
      tensor_type_names.push_back(type_name);
    }
  }
  JsonArray tensor_types;
  for (const std::string &type_name : tensor_type_names) {
    tensor_types.push_back(JsonValue(type_name));
  }
  const bool has_token_embeddings = find_tensor_by_name(model.tensors, {"token_embd.weight", "tok_embeddings.weight"}) != nullptr;
  const bool has_output = find_tensor_by_name(model.tensors, {"output.weight", "lm_head.weight"}) != nullptr;
  const bool output_tied = !has_output && has_token_embeddings;
  JsonObject shared = {
    {"tokenEmbeddings", JsonValue(has_token_embeddings)},
    {"outputNorm", JsonValue(find_tensor_by_name(model.tensors, {"output_norm.weight", "norm.weight"}) != nullptr)},
    {"output", JsonValue(has_output || output_tied)},
    {"outputTied", JsonValue(output_tied)},
    {"family", JsonValue(spec.family)},
  };
  const std::uint64_t total_layers = model.block_count.value_or(0);
  std::uint64_t complete_layers = 0;
  JsonArray incomplete_layers;
  for (std::uint64_t layer = 0; layer < total_layers; layer += 1) {
    const std::vector<std::string> missing_labels = family_layer_missing_labels(model, spec, layer);
    if (missing_labels.empty()) {
      complete_layers += 1;
      continue;
    }
    JsonArray missing;
    for (const std::string &label : missing_labels) {
      missing.push_back(JsonValue(label));
    }
    incomplete_layers.push_back(JsonValue(JsonObject {
      {"layer", JsonValue(layer)},
      {"missing", JsonValue(missing)},
    }));
  }
  const bool shared_ready = json_bool_or_default(shared, "tokenEmbeddings", false)
    && json_bool_or_default(shared, "outputNorm", false)
    && json_bool_or_default(shared, "output", false);
  return {
    {"shared", JsonValue(shared)},
    {"tensorTypes", JsonValue(tensor_types)},
    {"totalLayers", JsonValue(total_layers)},
    {"completeLayers", JsonValue(complete_layers)},
    {"incompleteLayers", JsonValue(incomplete_layers)},
    {"tensorDirectoryComplete", JsonValue(shared_ready && (total_layers == 0 || complete_layers == total_layers))},
  };
}

// ---- Vision tower (Phase 2 of .agent/vision/01_native_vision_plan.md) -------
//
// Gemma 3n ships the vision tower inside the SAME GGUF as the language model -
// there is no separate mmproj to load. The tensors are named `v.*` beside the
// language model's `blk.*`, and the block is an ordinary ViT block: attention,
// a gated FFN, and two norms. That is the same shape the language path already
// runs, which is why this is an indexing problem before it is a maths problem.
//
//   v.patch_embd.weight        [patch, patch, channels, hidden]
//   v.position_embd.weight     position table
//   v.blk.N.{ln1,attn_*,ffn_*} the tower proper
//   mm.input_projection.weight [vision hidden, language hidden]
//
// Models that split the tower into mmproj-*.gguf (LLaVA and friends) will need
// a second file loaded into this same index; the index is shaped for that so
// the rest of the pipeline never learns which layout it came from.
// The block carries PRE and POST norms around both sub-layers, and per-head
// query/key norms - Gemma's shape, not vanilla ViT's. attn_q_norm/attn_k_norm
// are head_dim wide (64 against a 768 hidden, so 12 heads), which is also how
// the head count is recovered without a metadata key.
struct VisionLayerTensorIndex {
  std::size_t ln1 = k_missing_tensor_index;
  std::size_t ln1_bias = k_missing_tensor_index;
  std::size_t ln2 = k_missing_tensor_index;
  std::size_t ln2_bias = k_missing_tensor_index;
  std::size_t attn_q = k_missing_tensor_index;
  std::size_t attn_k = k_missing_tensor_index;
  std::size_t attn_v = k_missing_tensor_index;
  std::size_t attn_out = k_missing_tensor_index;
  std::size_t attn_q_bias = k_missing_tensor_index;
  std::size_t attn_k_bias = k_missing_tensor_index;
  std::size_t attn_v_bias = k_missing_tensor_index;
  std::size_t attn_out_bias = k_missing_tensor_index;
  std::size_t attn_q_norm = k_missing_tensor_index;
  std::size_t attn_k_norm = k_missing_tensor_index;
  std::size_t attn_post_norm = k_missing_tensor_index;
  std::size_t ffn_post_norm = k_missing_tensor_index;
  std::size_t ffn_gate = k_missing_tensor_index;
  std::size_t ffn_up = k_missing_tensor_index;
  std::size_t ffn_up_bias = k_missing_tensor_index;
  std::size_t ffn_down = k_missing_tensor_index;
  std::size_t ffn_down_bias = k_missing_tensor_index;

  // ffn_gate is deliberately NOT required: a gate means a gated MLP, and its
  // absence means a plain one. CLIP and SigLIP towers have no gate, and
  // demanding one rejected them as "incomplete" when they were merely a
  // different family.
  bool complete() const {
    return ln1 != k_missing_tensor_index
      && attn_q != k_missing_tensor_index
      && attn_k != k_missing_tensor_index
      && attn_v != k_missing_tensor_index
      && attn_out != k_missing_tensor_index
      && ffn_up != k_missing_tensor_index
      && ffn_down != k_missing_tensor_index;
  }
};

// How a vision tower is put together, so one forward pass serves every family
// instead of a branch per model.
//
// Resolved from TENSOR PRESENCE, not an architecture allow-list. Vision GGUFs
// come from many exporters and the arch string is unreliable (gemma-3's mmproj
// declares `clip`), but the tensors cannot lie: a bias tensor means LayerNorm,
// an ffn_gate means a gated MLP, an attn_q_norm means per-head QK norms. New
// families that reuse these building blocks then work without code changes,
// and genuinely new ones fail loudly at detection rather than quietly at maths.
struct VisionFamilySpec {
  // LayerNorm (has bias, subtracts the mean) vs RMSNorm (no bias).
  bool layer_norm = false;
  // Gated MLP: silu(gate) * up. Otherwise a plain up -> activation -> down.
  bool gated_ffn = false;
  // Per-head query/key norms, as Gemma's inline tower has.
  bool qk_norms = false;
  // Post-norms around each sub-layer's output, as opposed to pre-norm only.
  bool post_norms = false;
  // Projection biases (q/k/v/out and the MLP), present on CLIP/SigLIP towers.
  bool projection_biases = false;
  // A norm applied to the tower output before the projector (mm.soft_emb_norm).
  bool soft_embedding_norm = false;
  // GELU for CLIP/SigLIP MLPs, SiLU for gated ones.
  bool gelu_activation = false;
  double norm_epsilon = 1e-6;
  // The square edge the tower expects, for Phase 3 preprocessing.
  std::uint64_t image_size = 0;
  // Average-pool factor applied to the patch GRID before projection, so an
  // image costs a fixed number of prompt positions regardless of its patch
  // count. gemma-3 pools 4x4 (64x64 patches -> 16x16 = 256 tokens); Gemma 3n
  // declares 3. 1 means no pooling.
  //
  // This is not cosmetic: skipping it hands the language model 4096 positions
  // where it was trained to see 256, which is a different input entirely.
  std::uint64_t pool_factor = 1;
  std::string projector_type;
  std::string label = "unknown";
};

struct VisionTensorIndex {
  std::size_t patch_embedding = k_missing_tensor_index;
  std::size_t patch_embedding_bias = k_missing_tensor_index;
  std::size_t position_embedding = k_missing_tensor_index;
  std::size_t input_projection = k_missing_tensor_index;
  std::size_t soft_embedding_norm = k_missing_tensor_index;
  std::size_t post_ln_weight = k_missing_tensor_index;
  std::size_t post_ln_bias = k_missing_tensor_index;
  VisionFamilySpec family;
  std::vector<VisionLayerTensorIndex> layers;
  std::uint64_t patch_size = 0;
  std::uint64_t channels = 0;
  std::uint64_t hidden_size = 0;
  std::uint64_t language_hidden_size = 0;
  std::uint64_t head_dim = 0;
  std::uint64_t head_count = 0;
  // Whether mm.input_projection is stored [vision, language] (true) or
  // [language, vision] (false). Exporters differ; see the resolver.
  bool projection_is_vision_major = true;
  std::vector<std::string> missing;
  bool ready = false;
};

// Vision metadata lives under a prefix that depends on the exporter:
// `clip.vision.*` in a standalone mmproj, `<arch>.vision.*` when the tower is
// packed beside the language model. Same keys, different prefix - so try both
// rather than special-casing a family.
static double vision_metadata_number(
  const ParsedGgufModel &model,
  const std::string &suffix,
  double fallback) {
  const std::vector<std::string> prefixes {
    "clip.vision.",
    model.architecture + ".vision.",
    "clip.",
    model.architecture + ".",
  };
  for (const std::string &prefix : prefixes) {
    const JsonValue *found = find_object_value(model.metadata, prefix + suffix);
    if (found != nullptr && std::holds_alternative<double>(found->value)) {
      return std::get<double>(found->value);
    }
  }
  return fallback;
}

static std::string vision_metadata_text(const ParsedGgufModel &model, const std::string &suffix) {
  for (const std::string &prefix : { std::string("clip."), model.architecture + "." }) {
    const JsonValue *found = find_object_value(model.metadata, prefix + suffix);
    if (found != nullptr && std::holds_alternative<std::string>(found->value)) {
      return std::get<std::string>(found->value);
    }
  }
  return std::string();
}

static bool vision_metadata_flag(const ParsedGgufModel &model, const std::string &suffix, bool fallback) {
  for (const std::string &prefix : { std::string("clip."), model.architecture + "." }) {
    const JsonValue *found = find_object_value(model.metadata, prefix + suffix);
    if (found != nullptr && std::holds_alternative<bool>(found->value)) {
      return std::get<bool>(found->value);
    }
  }
  return fallback;
}

static VisionTensorIndex build_vision_tensor_index(const ParsedGgufModel &model) {
  VisionTensorIndex index;
  index.patch_embedding = find_tensor_index_by_name(model.tensors, {"v.patch_embd.weight"});
  index.position_embedding = find_tensor_index_by_name(model.tensors, {"v.position_embd.weight"});
  index.input_projection = find_tensor_index_by_name(model.tensors, {"mm.input_projection.weight"});

  if (index.patch_embedding == k_missing_tensor_index) {
    index.missing.push_back("v.patch_embd.weight");
  } else {
    // [patch, patch, channels, hidden] - the conv kernel that turns a 16x16
    // RGB tile into one hidden-width row.
    const std::vector<std::uint64_t> &dims = model.tensors[index.patch_embedding].dimensions;
    if (dims.size() >= 4) {
      index.patch_size = dims[0];
      index.channels = dims[2];
      index.hidden_size = dims[3];
    }
  }
  if (index.input_projection == k_missing_tensor_index) {
    index.missing.push_back("mm.input_projection.weight");
  } else {
    // Exporters disagree on the projector's orientation: Gemma's inline tower
    // stores [vision, language] and the gemma-3 mmproj stores [language,
    // vision]. Rather than guess, anchor on the width we already know for
    // certain - the vision hidden size from the patch kernel - and take the
    // OTHER dimension as the language width.
    const std::vector<std::uint64_t> &dims = model.tensors[index.input_projection].dimensions;
    if (dims.size() >= 2) {
      if (index.hidden_size != 0 && dims[0] == index.hidden_size) {
        index.language_hidden_size = dims[1];
        index.projection_is_vision_major = true;
      } else if (index.hidden_size != 0 && dims[1] == index.hidden_size) {
        index.language_hidden_size = dims[0];
        index.projection_is_vision_major = false;
      } else {
        index.missing.push_back("mm.input_projection.weight (neither dimension matches the vision hidden size)");
      }
    }
  }

  // Walk blocks until one is absent. Counting this way rather than trusting a
  // metadata key means a truncated or partially-quantized tower reports the
  // depth it actually has.
  for (std::size_t block = 0;; block += 1) {
    const std::string prefix = "v.blk." + std::to_string(block) + ".";
    VisionLayerTensorIndex layer;
    layer.ln1 = find_tensor_index_by_name(model.tensors, {prefix + "ln1.weight"});
    layer.attn_q = find_tensor_index_by_name(model.tensors, {prefix + "attn_q.weight"});
    layer.attn_k = find_tensor_index_by_name(model.tensors, {prefix + "attn_k.weight"});
    layer.attn_v = find_tensor_index_by_name(model.tensors, {prefix + "attn_v.weight"});
    // Gemma names it attn_out; other towers use attn_output. Accept both so a
    // split mmproj from a different family indexes through the same code.
    layer.attn_out = find_tensor_index_by_name(
      model.tensors, {prefix + "attn_out.weight", prefix + "attn_output.weight"});
    layer.attn_q_norm = find_tensor_index_by_name(model.tensors, {prefix + "attn_q_norm.weight"});
    layer.attn_k_norm = find_tensor_index_by_name(model.tensors, {prefix + "attn_k_norm.weight"});
    layer.attn_post_norm = find_tensor_index_by_name(model.tensors, {prefix + "attn_post_norm.weight"});
    layer.ffn_post_norm = find_tensor_index_by_name(model.tensors, {prefix + "ffn_post_norm.weight"});
    // CLIP/SigLIP towers carry a second pre-norm and biases everywhere; Gemma's
    // inline tower carries neither. Both are read, and the family spec below
    // decides which the forward pass uses.
    layer.ln1_bias = find_tensor_index_by_name(model.tensors, {prefix + "ln1.bias"});
    layer.ln2 = find_tensor_index_by_name(model.tensors, {prefix + "ln2.weight"});
    layer.ln2_bias = find_tensor_index_by_name(model.tensors, {prefix + "ln2.bias"});
    layer.attn_q_bias = find_tensor_index_by_name(model.tensors, {prefix + "attn_q.bias"});
    layer.attn_k_bias = find_tensor_index_by_name(model.tensors, {prefix + "attn_k.bias"});
    layer.attn_v_bias = find_tensor_index_by_name(model.tensors, {prefix + "attn_v.bias"});
    layer.attn_out_bias = find_tensor_index_by_name(
      model.tensors, {prefix + "attn_out.bias", prefix + "attn_output.bias"});
    layer.ffn_up_bias = find_tensor_index_by_name(model.tensors, {prefix + "ffn_up.bias"});
    layer.ffn_down_bias = find_tensor_index_by_name(model.tensors, {prefix + "ffn_down.bias"});
    layer.ffn_gate = find_tensor_index_by_name(model.tensors, {prefix + "ffn_gate.weight"});
    layer.ffn_up = find_tensor_index_by_name(model.tensors, {prefix + "ffn_up.weight"});
    layer.ffn_down = find_tensor_index_by_name(model.tensors, {prefix + "ffn_down.weight"});
    if (layer.ln1 == k_missing_tensor_index && layer.attn_q == k_missing_tensor_index) {
      break;
    }
    if (!layer.complete()) {
      index.missing.push_back(prefix + "(incomplete block)");
      break;
    }
    index.layers.push_back(layer);
  }

  index.patch_embedding_bias = find_tensor_index_by_name(model.tensors, {"v.patch_embd.bias"});
  index.soft_embedding_norm = find_tensor_index_by_name(model.tensors, {"mm.soft_emb_norm.weight"});
  index.post_ln_weight = find_tensor_index_by_name(model.tensors, {"v.post_ln.weight"});
  index.post_ln_bias = find_tensor_index_by_name(model.tensors, {"v.post_ln.bias"});

  if (index.layers.empty()) {
    index.missing.push_back("v.blk.*");
  } else {
    // ---- resolve the family from what block 0 actually contains ----
    const VisionLayerTensorIndex &probe = index.layers.front();
    VisionFamilySpec &family = index.family;
    family.layer_norm = probe.ln1_bias != k_missing_tensor_index;
    family.gated_ffn = probe.ffn_gate != k_missing_tensor_index;
    family.qk_norms = probe.attn_q_norm != k_missing_tensor_index;
    family.post_norms = probe.attn_post_norm != k_missing_tensor_index
      || probe.ffn_post_norm != k_missing_tensor_index;
    family.projection_biases = probe.attn_q_bias != k_missing_tensor_index;
    family.soft_embedding_norm = index.soft_embedding_norm != k_missing_tensor_index;
    // GELU vs SiLU for the non-gated MLP. Declared by CLIP exporters; gated
    // towers use SiLU by construction, so the default follows the gate.
    family.gelu_activation = vision_metadata_flag(model, "use_gelu", !family.gated_ffn);
    family.norm_epsilon = vision_metadata_number(
      model, "attention.layer_norm_epsilon",
      vision_metadata_number(model, "attention.layer_norm_rms_epsilon", 1e-6));
    // Phase 3 will need these; recorded here so preprocessing never has to
    // re-open the file or guess.
    family.image_size = static_cast<std::uint64_t>(vision_metadata_number(model, "image_size", 0.0));
    family.projector_type = vision_metadata_text(model, "projector_type");
    // Declared where the exporter bothered (Gemma 3n says 3). gemma-3's mmproj
    // does not declare it - llama.cpp hardcodes 4 there - so that one case is
    // named explicitly rather than defaulted, and anything unrecognised pools
    // not at all rather than pooling wrongly.
    const double declared_scale = vision_metadata_number(model, "projector.scale_factor", 0.0);
    if (declared_scale >= 1.0) {
      family.pool_factor = static_cast<std::uint64_t>(declared_scale);
    } else if (family.projector_type == "gemma3") {
      family.pool_factor = 4;
    }
    // A label for humans reading a probe, never a control-flow input: the
    // booleans above are what the forward pass reads, so an unrecognised
    // combination still runs on its true shape rather than a guessed family.
    if (family.layer_norm && !family.gated_ffn) {
      family.label = family.soft_embedding_norm ? "siglip/gemma-3-mmproj" : "clip-vit";
    } else if (!family.layer_norm && family.gated_ffn && family.qk_norms) {
      family.label = "gemma-inline";
    } else if (!family.layer_norm && family.gated_ffn) {
      family.label = "rmsnorm-gated";
    } else {
      family.label = "mixed";
    }
  }
  if (!index.layers.empty()) {
    // Head count, two independent routes. Metadata is authoritative when it is
    // there; the query-norm width is the fallback for towers that omit the key.
    // Both are tried and cross-checked, because they agree on Gemma's inline
    // tower (12 either way) and only one exists on SigLIP.
    const VisionLayerTensorIndex &first = index.layers.front();
    const double declared = vision_metadata_number(model, "attention.head_count", 0.0);
    if (declared > 0.0) {
      index.head_count = static_cast<std::uint64_t>(declared);
      if (index.head_count > 0) {
        index.head_dim = index.hidden_size / index.head_count;
      }
    }
    if (first.attn_q_norm != k_missing_tensor_index) {
      const std::vector<std::uint64_t> &dims = model.tensors[first.attn_q_norm].dimensions;
      if (!dims.empty() && dims[0] > 0) {
        const std::uint64_t derived_dim = dims[0];
        const std::uint64_t derived_count = index.hidden_size / derived_dim;
        if (index.head_count != 0 && index.head_count != derived_count) {
          // Disagreement means one of the two readings is wrong, and guessing
          // which would be exactly the silent-wrongness this plan guards
          // against. Refuse instead.
          index.missing.push_back(
            "attention.head_count disagrees with the query-norm width ("
            + std::to_string(index.head_count) + " vs " + std::to_string(derived_count) + ")");
        }
        index.head_count = derived_count;
        index.head_dim = derived_dim;
      }
    }
  }
  index.ready = index.missing.empty()
    && index.hidden_size > 0
    && index.language_hidden_size > 0
    && index.head_count > 0;
  return index;
}

static LlamaTensorIndex build_llama_tensor_index(const ParsedGgufModel &model) {
  LlamaTensorIndex index;
  const std::optional<FamilySpec> family_spec = resolve_family_spec(model);
  index.token_embeddings = find_tensor_index_by_name(model.tensors, {"token_embd.weight", "tok_embeddings.weight"});
  index.output_norm = find_tensor_index_by_name(model.tensors, {"output_norm.weight", "norm.weight"});
  index.output = find_tensor_index_by_name(model.tensors, {"output.weight", "lm_head.weight"});
  index.output_norm_bias = find_tensor_index_by_name(model.tensors, {"output_norm.bias", "norm.bias"});
  index.output_bias = find_tensor_index_by_name(model.tensors, {"output.bias", "lm_head.bias"});
  index.rope_factors_long = find_tensor_index_by_name(model.tensors, {"rope_factors_long.weight", "blk.0.rope_factors_long.weight"});
  index.rope_factors_short = find_tensor_index_by_name(model.tensors, {"rope_factors_short.weight", "blk.0.rope_factors_short.weight"});
  index.rope_freqs = find_tensor_index_by_name(model.tensors, {"rope_freqs.weight"});
  index.per_layer_token_embeddings = find_tensor_index_by_name(model.tensors, {"per_layer_token_embd.weight"});
  index.per_layer_model_projection = find_tensor_index_by_name(model.tensors, {"per_layer_model_proj.weight"});
  index.per_layer_projection_norm = find_tensor_index_by_name(model.tensors, {"per_layer_proj_norm.weight"});
  if (index.output == k_missing_tensor_index && index.token_embeddings != k_missing_tensor_index) {
    index.output = index.token_embeddings;
    index.output_tied = true;
  }

  if (index.token_embeddings == k_missing_tensor_index) {
    index.missing_shared.push_back("tokenEmbeddings");
  }
  if (index.output_norm == k_missing_tensor_index) {
    index.missing_shared.push_back("outputNorm");
  }
  if (index.output == k_missing_tensor_index) {
    index.missing_shared.push_back("output");
  }
  const bool requires_per_layer_inputs = is_gemma_architecture(model.architecture)
    && model.embedding_length_per_layer_input.value_or(0) > 0;
  if (requires_per_layer_inputs && index.per_layer_token_embeddings == k_missing_tensor_index) {
    index.missing_shared.push_back("perLayerTokenEmbeddings");
  }
  if (requires_per_layer_inputs && index.per_layer_model_projection == k_missing_tensor_index) {
    index.missing_shared.push_back("perLayerModelProjection");
  }
  if (requires_per_layer_inputs && index.per_layer_projection_norm == k_missing_tensor_index) {
    index.missing_shared.push_back("perLayerProjectionNorm");
  }

  const std::vector<std::string> dequant_types = {
    "F32", "F16", "BF16", "Q8_0", "Q4_K", "Q5_K", "Q6_K",
    "Q4_0", "Q2_K", "Q3_K", "IQ4_XS", "TQ1_0", "TQ2_0",
  };
  for (const std::string &type_name : dequant_types) {
    if (std::any_of(model.tensors.begin(), model.tensors.end(), [&](const TensorInfo &tensor) {
      return tensor.ggml_type_name == type_name;
    })) {
      index.supported_dequant_types.push_back(JsonValue(type_name));
    }
  }

  const std::uint64_t total_layers = model.block_count.value_or(0);
  index.layers.resize(static_cast<std::size_t>(total_layers));
  for (std::uint64_t layer = 0; layer < total_layers; layer += 1) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    LlamaLayerTensorIndex &layer_index = index.layers[static_cast<std::size_t>(layer)];
    layer_index.attention_norm = find_tensor_index_by_name(model.tensors, {prefix + "attn_norm.weight"});
    layer_index.query = find_tensor_index_by_name(model.tensors, {prefix + "attn_q.weight"});
    layer_index.key = find_tensor_index_by_name(model.tensors, {prefix + "attn_k.weight"});
    layer_index.value = find_tensor_index_by_name(model.tensors, {prefix + "attn_v.weight"});
    layer_index.output = find_tensor_index_by_name(model.tensors, {prefix + "attn_output.weight"});
    layer_index.query_norm = find_tensor_index_by_name(model.tensors, {prefix + "attn_q_norm.weight"});
    layer_index.key_norm = find_tensor_index_by_name(model.tensors, {prefix + "attn_k_norm.weight"});
    layer_index.ffn_norm = find_tensor_index_by_name(model.tensors, {prefix + "ffn_norm.weight"});
    layer_index.gate = find_tensor_index_by_name(model.tensors, {prefix + "ffn_gate.weight"});
    layer_index.down = find_tensor_index_by_name(model.tensors, {prefix + "ffn_down.weight"});
    layer_index.up = find_tensor_index_by_name(model.tensors, {prefix + "ffn_up.weight"});
    layer_index.post_attention_norm = find_tensor_index_by_name(model.tensors, {prefix + "post_attention_norm.weight"});
    layer_index.post_ffw_norm = find_tensor_index_by_name(model.tensors, {prefix + "post_ffw_norm.weight"});
    layer_index.post_norm = find_tensor_index_by_name(model.tensors, {prefix + "post_norm.weight"});
    layer_index.per_layer_input_gate = find_tensor_index_by_name(model.tensors, {prefix + "inp_gate.weight"});
    layer_index.per_layer_projection = find_tensor_index_by_name(model.tensors, {prefix + "proj.weight"});
    layer_index.layer_output_scale = find_tensor_index_by_name(model.tensors, {prefix + "layer_output_scale.weight"});
    layer_index.qkv_fused = find_tensor_index_by_name(model.tensors, {prefix + "attn_qkv.weight"});
    layer_index.qkv_fused_bias = find_tensor_index_by_name(model.tensors, {prefix + "attn_qkv.bias"});
    layer_index.attention_norm_bias = find_tensor_index_by_name(model.tensors, {prefix + "attn_norm.bias"});
    layer_index.query_bias = find_tensor_index_by_name(model.tensors, {prefix + "attn_q.bias"});
    layer_index.key_bias = find_tensor_index_by_name(model.tensors, {prefix + "attn_k.bias"});
    layer_index.value_bias = find_tensor_index_by_name(model.tensors, {prefix + "attn_v.bias"});
    layer_index.output_bias = find_tensor_index_by_name(model.tensors, {prefix + "attn_output.bias"});
    layer_index.ffn_norm_bias = find_tensor_index_by_name(model.tensors, {prefix + "ffn_norm.bias"});
    layer_index.up_bias = find_tensor_index_by_name(model.tensors, {prefix + "ffn_up.bias"});
    layer_index.down_bias = find_tensor_index_by_name(model.tensors, {prefix + "ffn_down.bias"});
    layer_index.moe_gate_inp = find_tensor_index_by_name(model.tensors, {prefix + "ffn_gate_inp.weight"});
    layer_index.moe_gate_exps = find_tensor_index_by_name(model.tensors, {prefix + "ffn_gate_exps.weight"});
    layer_index.moe_up_exps = find_tensor_index_by_name(model.tensors, {prefix + "ffn_up_exps.weight"});
    layer_index.moe_down_exps = find_tensor_index_by_name(model.tensors, {prefix + "ffn_down_exps.weight"});
    layer_index.shexp_gate_inp = find_tensor_index_by_name(model.tensors, {prefix + "ffn_gate_inp_shexp.weight"});
    layer_index.shexp_gate = find_tensor_index_by_name(model.tensors, {prefix + "ffn_gate_shexp.weight"});
    layer_index.shexp_up = find_tensor_index_by_name(model.tensors, {prefix + "ffn_up_shexp.weight"});
    layer_index.shexp_down = find_tensor_index_by_name(model.tensors, {prefix + "ffn_down_shexp.weight"});

    if (family_spec.has_value()) {
      // Family-spec models report gaps against their own graph requirements;
      // the llama/gemma requirement lists below stay untouched for those paths.
      JsonArray family_missing;
      for (const std::string &label : family_layer_missing_labels(model, family_spec.value(), layer)) {
        family_missing.push_back(JsonValue(label));
      }
      if (!family_missing.empty()) {
        index.missing_layers.push_back(JsonValue(JsonObject {
          {"layer", JsonValue(layer)},
          {"missing", JsonValue(family_missing)},
        }));
      }
      continue;
    }

    JsonArray missing;
    if (layer_index.attention_norm == k_missing_tensor_index) missing.push_back(JsonValue("attentionNorm"));
    if (layer_index.query == k_missing_tensor_index) missing.push_back(JsonValue("query"));
    if (layer_index.key == k_missing_tensor_index) missing.push_back(JsonValue("key"));
    if (layer_index.value == k_missing_tensor_index) missing.push_back(JsonValue("value"));
    if (layer_index.output == k_missing_tensor_index) missing.push_back(JsonValue("output"));
    if (layer_index.ffn_norm == k_missing_tensor_index) missing.push_back(JsonValue("ffnNorm"));
    if (layer_index.gate == k_missing_tensor_index) missing.push_back(JsonValue("gate"));
    if (layer_index.down == k_missing_tensor_index) missing.push_back(JsonValue("down"));
    if (layer_index.up == k_missing_tensor_index) missing.push_back(JsonValue("up"));
    if (is_gemma_architecture(model.architecture)) {
      if (layer_index.query_norm == k_missing_tensor_index) missing.push_back(JsonValue("queryNorm"));
      if (layer_index.key_norm == k_missing_tensor_index) missing.push_back(JsonValue("keyNorm"));
      if (layer_index.post_attention_norm == k_missing_tensor_index) missing.push_back(JsonValue("postAttentionNorm"));
      if (layer_index.post_ffw_norm == k_missing_tensor_index) missing.push_back(JsonValue("postFfwNorm"));
      if (requires_per_layer_inputs && layer_index.per_layer_input_gate == k_missing_tensor_index) missing.push_back(JsonValue("perLayerInputGate"));
      if (requires_per_layer_inputs && layer_index.per_layer_projection == k_missing_tensor_index) missing.push_back(JsonValue("perLayerProjection"));
      if (requires_per_layer_inputs && layer_index.post_norm == k_missing_tensor_index) missing.push_back(JsonValue("perLayerPostNorm"));
      if (requires_per_layer_inputs && layer_index.layer_output_scale == k_missing_tensor_index) missing.push_back(JsonValue("layerOutputScale"));
    }
    if (!missing.empty()) {
      index.missing_layers.push_back(JsonValue(JsonObject {
        {"layer", JsonValue(layer)},
        {"missing", JsonValue(missing)},
      }));
    }
  }

  index.ready = index.missing_shared.empty() && index.missing_layers.empty();
  return index;
}

static std::optional<double> infer_parameter_estimate(const ParsedGgufModel &model) {
  if (!model.embedding_length.has_value() || !model.block_count.has_value() || model.file_size_bytes == 0) {
    return std::nullopt;
  }
  double bits_per_weight = 4.5;
  if (model.quantization == "q8_0") {
    bits_per_weight = 8.0;
  } else if (model.quantization == "f16") {
    bits_per_weight = 16.0;
  } else if (model.quantization == "f32") {
    bits_per_weight = 32.0;
  } else if (model.quantization == "q6_k") {
    bits_per_weight = 6.4;
  } else if (model.quantization == "q5_k_m") {
    bits_per_weight = 5.5;
  }
  const double estimate = (static_cast<double>(model.file_size_bytes) * 8.0) / bits_per_weight / 1000000000.0;
  return std::round(estimate * 100.0) / 100.0;
}

static ParsedGgufModel parse_gguf(FileView &file_view, const std::filesystem::path &file_path) {
  if (file_view.data() == nullptr || file_view.size() < 4) {
    throw WorkerError("NOT_GGUF", "File is too small to contain a valid GGUF header.");
  }
  ParsedGgufModel model;
  model.file_path = file_path;
  model.file_name = file_path.filename().string();
  model.file_size_bytes = file_view.size();
  model.memory_mode = file_view.mode();
  model.pinned_memory = file_view.pin_applied();
  model.warnings = file_view.warnings();

  ByteReader reader(file_view.data(), file_view.size());
  std::string magic(reinterpret_cast<const char *>(file_view.data()), 4);
  if (magic != "GGUF") {
    throw WorkerError("NOT_GGUF", "File does not start with the GGUF magic header.");
  }
  reader.read_u32();
  model.version = reader.read_u32();
  model.tensor_count = reader.read_u64();
  model.metadata_kv_count = reader.read_u64();

  for (std::uint64_t index = 0; index < model.metadata_kv_count; index += 1) {
    const std::string key = reader.read_string();
    const auto type = static_cast<GgufValueType>(reader.read_u32());
    model.metadata.emplace(key, read_metadata_value(reader, type));
  }

  model.tensors.reserve(static_cast<std::size_t>(model.tensor_count));
  for (std::uint64_t index = 0; index < model.tensor_count; index += 1) {
    TensorInfo tensor;
    tensor.name = reader.read_string();
    const std::uint32_t dimension_count = reader.read_u32();
    tensor.dimensions.reserve(dimension_count);
    for (std::uint32_t dimension_index = 0; dimension_index < dimension_count; dimension_index += 1) {
      tensor.dimensions.push_back(reader.read_u64());
    }
    tensor.ggml_type = reader.read_u32();
    tensor.offset = reader.read_u64();
    model.tensors.push_back(std::move(tensor));
  }

  const std::uint64_t alignment = json_to_u64(
    lookup_first(model.metadata, {"general.alignment"}, {".alignment"})).value_or(32);
  model.tensor_data_offset = align_offset(static_cast<std::uint64_t>(reader.offset()), alignment);
  for (TensorInfo &tensor : model.tensors) {
    tensor.element_count = tensor_element_count(tensor.dimensions);
    tensor.absolute_offset = model.tensor_data_offset + tensor.offset;
    if (const auto layout = ggml_type_layout(tensor.ggml_type); layout.has_value()) {
      tensor.ggml_type_name = layout->name;
      const std::uint64_t block_count = (tensor.element_count + layout->block_size - 1) / layout->block_size;
      tensor.byte_length = block_count * layout->bytes_per_block;
      model.tensor_data_bytes = std::max(model.tensor_data_bytes, tensor.offset + tensor.byte_length);
      if (tensor.absolute_offset + tensor.byte_length > model.file_size_bytes) {
        model.warnings.push_back(
          "Tensor '" + tensor.name + "' extends beyond the GGUF file size; tensor data may be incomplete.");
      }
    } else {
      model.warnings.push_back("Unsupported GGML tensor type " + std::to_string(tensor.ggml_type) + " for tensor '" + tensor.name + "'.");
    }
  }

  model.metadata_hash = fnv1a_64_hex(file_view.data(), reader.offset());

  model.architecture = json_to_string(
    lookup_first(model.metadata, {"general.architecture"}, {".architecture"}));
  model.quantization = infer_quantization_from_file_type(
    json_to_u64(lookup_first(model.metadata, {"general.file_type", "general.quantization_version"}, {".file_type"})).value_or(0));
  if (model.quantization.empty()) {
    model.quantization = infer_quantization_from_filename(model.file_name);
  }
  model.tokenizer = json_to_string(
    lookup_first(model.metadata, {"tokenizer.ggml.model", "tokenizer.model"}, {".tokenizer_model"}));
  model.chat_template = json_to_string(
    lookup_first(model.metadata, {"tokenizer.chat_template", "general.chat_template"}, {".chat_template"}));
  model.context_length = json_to_u64(
    lookup_first(model.metadata, {"llama.context_length", "gemma.context_length", "gemma2.context_length", "gemma3.context_length", "gemma4.context_length", "general.context_length"}, {".context_length"}));
  model.embedding_length = json_to_u64(
    lookup_first(model.metadata, {"llama.embedding_length", "gemma.embedding_length", "gemma2.embedding_length", "gemma3.embedding_length", "gemma4.embedding_length", "general.embedding_length"}, {".embedding_length"}));
  model.block_count = json_to_u64(
    lookup_first(model.metadata, {"llama.block_count", "gemma.block_count", "gemma2.block_count", "gemma3.block_count", "gemma4.block_count", "general.block_count"}, {".block_count"}));
  model.attention_head_count = json_to_u64(
    lookup_first(model.metadata, {"llama.attention.head_count", "gemma.attention.head_count", "gemma2.attention.head_count", "gemma3.attention.head_count", "gemma4.attention.head_count", "general.attention.head_count"}, {".head_count"}));
  model.kv_head_count = json_to_u64(
    lookup_first(model.metadata, {"llama.attention.head_count_kv", "gemma.attention.head_count_kv", "gemma2.attention.head_count_kv", "gemma3.attention.head_count_kv", "gemma4.attention.head_count_kv", "general.attention.head_count_kv"}, {".head_count_kv"}));
  model.attention_key_length = json_to_u64(
    lookup_first(model.metadata, {"llama.attention.key_length", "gemma.attention.key_length", "gemma2.attention.key_length", "gemma3.attention.key_length", "gemma4.attention.key_length"}, {".attention.key_length"}));
  model.attention_value_length = json_to_u64(
    lookup_first(model.metadata, {"llama.attention.value_length", "gemma.attention.value_length", "gemma2.attention.value_length", "gemma3.attention.value_length", "gemma4.attention.value_length"}, {".attention.value_length"}));
  model.attention_key_length_swa = json_to_u64(
    lookup_first(model.metadata, {"gemma.attention.key_length_swa", "gemma2.attention.key_length_swa", "gemma3.attention.key_length_swa", "gemma4.attention.key_length_swa"}, {".attention.key_length_swa"}));
  model.attention_value_length_swa = json_to_u64(
    lookup_first(model.metadata, {"gemma.attention.value_length_swa", "gemma2.attention.value_length_swa", "gemma3.attention.value_length_swa", "gemma4.attention.value_length_swa"}, {".attention.value_length_swa"}));
  model.attention_sliding_window = json_to_u64(
    lookup_first(model.metadata, {"gemma.attention.sliding_window", "gemma2.attention.sliding_window", "gemma3.attention.sliding_window", "gemma4.attention.sliding_window"}, {".attention.sliding_window"}));
  model.attention_shared_kv_layers = json_to_u64(
    lookup_first(model.metadata, {"gemma3n.attention.shared_kv_layers", "gemma4.attention.shared_kv_layers"}, {".attention.shared_kv_layers"}));
  model.embedding_length_per_layer_input = json_to_u64(
    lookup_first(
      model.metadata,
      {
        "gemma3n.embedding_length_per_layer_input",
        "gemma4.embedding_length_per_layer_input",
      },
      {".embedding_length_per_layer_input"}));
  model.rope_freq_base = json_to_double(
    lookup_first(model.metadata, {"llama.rope.freq_base", "llama.rope.freq_base_train", "gemma.rope.freq_base", "gemma2.rope.freq_base", "gemma3.rope.freq_base", "gemma4.rope.freq_base"}, {".rope.freq_base"}));
  model.rope_freq_scale = json_to_double(
    lookup_first(model.metadata, {"llama.rope.scale_linear", "llama.rope.freq_scale_train"}, {".rope.scale_linear", ".rope.freq_scale"}));
  model.rope_freq_base_swa = json_to_double(
    lookup_first(model.metadata, {"gemma.rope.freq_base_swa", "gemma2.rope.freq_base_swa", "gemma3.rope.freq_base_swa", "gemma4.rope.freq_base_swa"}, {".rope.freq_base_swa"}));
  model.layer_norm_rms_epsilon = json_to_double(
    lookup_first(
      model.metadata,
      {
        "llama.attention.layer_norm_rms_epsilon",
        "gemma.attention.layer_norm_rms_epsilon",
        "gemma2.attention.layer_norm_rms_epsilon",
        "gemma3.attention.layer_norm_rms_epsilon",
        "gemma4.attention.layer_norm_rms_epsilon",
      },
      {".attention.layer_norm_rms_epsilon", ".layer_norm_rms_epsilon"}));
  model.final_logit_softcapping = json_to_double(
    lookup_first(
      model.metadata,
      {
        "gemma.final_logit_softcapping",
        "gemma2.final_logit_softcapping",
        "gemma3.final_logit_softcapping",
        "gemma4.final_logit_softcapping",
      },
      {".final_logit_softcapping"}));
  model.layer_norm_epsilon = json_to_double(
    lookup_first(model.metadata, {}, {".attention.layer_norm_epsilon"}));
  model.expert_count = json_to_u64(
    lookup_first(model.metadata, {}, {".expert_count"}));
  model.expert_used_count = json_to_u64(
    lookup_first(model.metadata, {}, {".expert_used_count"}));
  model.expert_shared_count = json_to_u64(
    lookup_first(model.metadata, {}, {".expert_shared_count"}));
  model.expert_weights_scale = json_to_double(
    lookup_first(model.metadata, {}, {".expert_weights_scale"}));
  model.expert_feed_forward_length = json_to_u64(
    lookup_first(model.metadata, {}, {".expert_feed_forward_length"}));
  model.expert_shared_feed_forward_length = json_to_u64(
    lookup_first(model.metadata, {}, {".expert_shared_feed_forward_length"}));
  model.leading_dense_block_count = json_to_u64(
    lookup_first(model.metadata, {}, {".leading_dense_block_count"}));
  model.rope_dimension_count = json_to_u64(
    lookup_first(model.metadata, {}, {".rope.dimension_count"}));
  model.rope_scaling_attn_factor = json_to_double(
    lookup_first(model.metadata, {}, {".rope.scaling.attn_factor"}));
  model.rope_scaling_original_context_length = json_to_u64(
    lookup_first(model.metadata, {}, {".rope.scaling.original_context_length"}));

  const std::vector<std::pair<std::string, std::string>> special_token_keys = {
    {"bos", "tokenizer.ggml.bos_token_id"},
    {"eos", "tokenizer.ggml.eos_token_id"},
    {"pad", "tokenizer.ggml.padding_token_id"},
    {"unk", "tokenizer.ggml.unknown_token_id"},
  };
  for (const auto &[label, key] : special_token_keys) {
    if (const auto value = json_to_u64(find_object_value(model.metadata, key)); value.has_value()) {
      model.special_tokens.emplace(label, JsonValue(value.value()));
    }
  }

  const std::vector<std::string> tokens = json_array_to_strings(
    lookup_first(model.metadata, {"tokenizer.ggml.tokens"}, {".tokens"}));
  const std::vector<std::string> merges = json_array_to_strings(
    lookup_first(model.metadata, {"tokenizer.ggml.merges"}, {".merges"}));
  model.token_vocabulary = tokens;
  model.token_merges = merges;
  model.token_vocabulary_size = static_cast<std::uint64_t>(tokens.size());
  const std::size_t preview_count = std::min<std::size_t>(8, tokens.size());
  for (std::size_t index = 0; index < preview_count; index += 1) {
    model.token_preview.push_back(JsonValue(tokens[index]));
  }
  model.token_merge_count = static_cast<std::uint64_t>(merges.size());
  const std::size_t merge_preview_count = std::min<std::size_t>(8, merges.size());
  for (std::size_t index = 0; index < merge_preview_count; index += 1) {
    model.merge_preview.push_back(JsonValue(merges[index]));
  }
  for (const auto &[label, token_id] : model.special_tokens) {
    if (!token_id.is_number()) {
      continue;
    }
    const std::uint64_t index = static_cast<std::uint64_t>(token_id.as_number());
    if (index < tokens.size()) {
      model.special_token_texts.emplace(label, JsonValue(tokens[static_cast<std::size_t>(index)]));
    }
  }
  if (const auto family_spec = resolve_family_spec(model); family_spec.has_value()) {
    model.executor_shape = analyze_family_executor_shape(model, family_spec.value());
  } else if (model.architecture == "llama" || is_gemma_architecture(model.architecture)) {
    model.executor_shape = analyze_llama_executor_shape(model);
  }

  if (model.architecture.empty()) {
    model.warnings.push_back("GGUF architecture metadata is missing.");
  }
  if (model.quantization.empty()) {
    model.warnings.push_back("GGUF quantization metadata is missing.");
  }
  if (!model.context_length.has_value()) {
    model.warnings.push_back("GGUF context length metadata is missing.");
  }

  model.parameter_estimate_b = infer_parameter_estimate(model);
  return model;
}

static JsonObject build_compatibility_object(const ParsedGgufModel &model) {
  const std::optional<FamilySpec> family_spec = resolve_family_spec(model);
  const char *blocked_reason = family_blocked_reason(model.architecture);
  const bool architecture_supported = model.architecture == "llama"
    || is_gemma_architecture(model.architecture)
    || family_spec.has_value();
  const bool quantization_supported = model.quantization == "f32"
    || model.quantization == "f16"
    || model.quantization == "bf16"
    || model.quantization == "q8_0"
    || model.quantization == "q4_k_m"
    || model.quantization == "q4_k_s"
    || model.quantization == "q5_k_m"
    || model.quantization == "q5_k_s"
    || model.quantization == "q6_k"
    || model.quantization == "q4_0"
    || model.quantization == "q2_k"
    || model.quantization == "q2_k_s"
    || model.quantization == "q3_k_s"
    || model.quantization == "q3_k_m"
    || model.quantization == "q3_k_l"
    || model.quantization == "iq4_xs"
    || model.quantization == "tq1_0"
    || model.quantization == "tq2_0";
  const JsonObject *executor_shape = model.executor_shape.empty() ? nullptr : &model.executor_shape;
  const bool tensor_directory_complete = executor_shape != nullptr
    && json_bool_or_default(*executor_shape, "tensorDirectoryComplete", false);
  bool tensor_types_supported = true;
  if (executor_shape != nullptr) {
    const JsonValue *tensor_types_value = find_object_value(*executor_shape, "tensorTypes");
    if (tensor_types_value != nullptr && tensor_types_value->is_array()) {
      for (const JsonValue &entry : tensor_types_value->as_array()) {
        const std::string type_name = entry.is_string() ? entry.as_string() : "";
        if (!type_name.empty()
          && type_name != "F32"
          && type_name != "F16"
          && type_name != "BF16"
          && type_name != "Q8_0"
          && type_name != "Q4_K"
          && type_name != "Q5_K"
          && type_name != "Q6_K"
          && type_name != "Q4_0"
          && type_name != "Q2_K"
          && type_name != "Q3_K"
          && type_name != "IQ4_XS"
          && type_name != "TQ1_0"
          && type_name != "TQ2_0") {
          tensor_types_supported = false;
          break;
        }
      }
    }
  }
  const bool zero_layer_graph_supported = architecture_supported
    && tensor_directory_complete
    && model.block_count.value_or(1) == 0
    && model.token_vocabulary_size > 0
    && tensor_types_supported;
  const std::uint64_t head_count = model.attention_head_count.value_or(0);
  const std::uint64_t kv_head_count = model.kv_head_count.value_or(head_count);
  const std::uint64_t hidden_size = model.embedding_length.value_or(0);
  // qwen3-style models declare an explicit head dim (attention.key_length)
  // that need not equal hidden/head_count; fall back to the classic division.
  const std::uint64_t effective_head_dim = model.attention_key_length.value_or(
    head_count > 0 && hidden_size % head_count == 0 ? hidden_size / head_count : 0);
  const bool layered_attention_shape_supported = model.block_count.value_or(0) > 0
    && hidden_size > 0
    && head_count > 0
    && kv_head_count > 0
    && head_count % kv_head_count == 0
    && effective_head_dim > 0
    && (effective_head_dim % 2 == 0);
  const bool minimal_graph_supported = architecture_supported
    && tensor_directory_complete
    && (zero_layer_graph_supported || layered_attention_shape_supported)
    && model.token_vocabulary_size > 0
    && tensor_types_supported;
  const bool tokenizer_metadata_supported = model.token_vocabulary_size > 0;
  std::string executor_reason;
  const std::string graph_label = family_spec.has_value()
    ? family_spec->family
    : (is_gemma_architecture(model.architecture) ? "Gemma" : "Llama");
  if (minimal_graph_supported) {
    executor_reason = model.block_count.value_or(0) == 0
      ? "This GGUF matches the native worker's minimal zero-layer execution subset."
      : "This GGUF matches the native worker's supported " + graph_label + " CPU execution subset.";
  } else if (!tensor_directory_complete) {
    executor_reason = "The compiled worker can load GGUF metadata, but this model is missing one or more required " + graph_label + " executor tensors in its directory.";
  } else if (!tensor_types_supported) {
    executor_reason = "The GGUF uses one or more tensor types outside the native worker's currently supported CPU dequantization subset.";
  } else if (!tokenizer_metadata_supported) {
    executor_reason = "The GGUF loads successfully, but it is missing tokenizer vocabulary metadata required for native prompt and token decoding.";
  } else if (model.block_count.value_or(0) > 0 && !layered_attention_shape_supported) {
    executor_reason = "The GGUF contains the core " + graph_label + " tensor layout, but its attention metadata or tensor shapes are outside the native worker's current execution subset.";
  } else {
    executor_reason = "The GGUF loads successfully, but its executor metadata is outside the native worker's current CPU execution subset.";
  }

  std::string architecture_reason;
  std::string architecture_status;
  if (architecture_supported) {
    architecture_status = "supported";
    architecture_reason = family_spec.has_value()
      ? "The native C++ worker executes the '" + family_spec->family + "' GGUF family natively."
      : "The native C++ worker can load this GGUF architecture's metadata and tensor directory.";
  } else if (blocked_reason != nullptr) {
    architecture_status = "blocked";
    architecture_reason = blocked_reason;
  } else {
    architecture_status = "unsupported";
    architecture_reason = "The native C++ worker accepts llama, gemma, qwen2, qwen3, mistral, mixtral, "
      "phi, phi3/phi4, qwen2moe, qwen3moe, deepseek, glm4, and starcoder2 GGUF families; this architecture is outside that set.";
  }
  JsonObject architecture = {
    {"key", JsonValue(model.architecture.empty() ? "unknown" : model.architecture)},
    {"status", JsonValue(architecture_status)},
    {"reason", JsonValue(architecture_reason)},
  };
  if (family_spec.has_value()) {
    architecture.emplace("family", JsonValue(family_spec->family));
  }
  JsonObject quantization = {
    {"key", JsonValue(model.quantization.empty() ? "unknown" : model.quantization)},
    {"status", JsonValue(quantization_supported ? "supported" : "unsupported")},
    {"reason", JsonValue(quantization_supported
      ? "The native C++ worker accepts f32, f16, bf16, q8_0, q4_0, q2_k, q3_k, q4_k, q5_k, q6_k, iq4_xs, tq1_0, and tq2_0 GGUF files at load time."
      : "The native C++ worker currently accepts only f32, f16, bf16, q8_0, q4_0, q2_k, q3_k, q4_k, q5_k, q6_k, iq4_xs, tq1_0, and tq2_0 GGUF files at load time.")},
  };
  JsonObject executor = {
    {"status", JsonValue(minimal_graph_supported ? "supported" : "unsupported")},
    {"reason", JsonValue(executor_reason)},
    {"tensorDirectoryComplete", JsonValue(tensor_directory_complete)},
    {"tensorTypesSupported", JsonValue(tensor_types_supported)},
    {"tokenizerMetadataSupported", JsonValue(tokenizer_metadata_supported)},
    {"zeroLayerSubsetSupported", JsonValue(zero_layer_graph_supported)},
    {"layeredAttentionShapeSupported", JsonValue(layered_attention_shape_supported)},
  };
  if (executor_shape != nullptr) {
    executor.emplace("shape", JsonValue(*executor_shape));
  }
  return {
    {"architecture", JsonValue(architecture)},
    {"quantization", JsonValue(quantization)},
    {"executor", JsonValue(executor)},
  };
}

static JsonValue build_tensors_preview(const ParsedGgufModel &model, std::size_t limit = 8) {
  JsonArray preview;
  const std::size_t count = std::min<std::size_t>(limit, model.tensors.size());
  preview.reserve(count);
  for (std::size_t index = 0; index < count; index += 1) {
    const TensorInfo &tensor = model.tensors[index];
    JsonArray dimensions;
    for (std::uint64_t dimension : tensor.dimensions) {
      dimensions.push_back(JsonValue(dimension));
    }
    JsonObject entry = {
      {"name", JsonValue(tensor.name)},
      {"dimensions", JsonValue(dimensions)},
      {"ggmlType", JsonValue(static_cast<std::uint64_t>(tensor.ggml_type))},
      {"offsetBytes", JsonValue(std::to_string(tensor.offset))},
      {"absoluteOffsetBytes", JsonValue(std::to_string(tensor.absolute_offset))},
    };
    if (!tensor.ggml_type_name.empty()) {
      entry.emplace("ggmlTypeName", JsonValue(tensor.ggml_type_name));
    }
    if (tensor.element_count > 0) {
      entry.emplace("elementCount", JsonValue(tensor.element_count));
    }
    if (tensor.byte_length > 0) {
      entry.emplace("byteLength", JsonValue(tensor.byte_length));
    }
    preview.push_back(JsonValue(entry));
  }
  return JsonValue(preview);
}

static JsonObject build_model_object(const ParsedGgufModel &model) {
  JsonObject rope_settings;
  if (model.rope_freq_base.has_value()) {
    rope_settings.emplace("freqBase", JsonValue(model.rope_freq_base.value()));
  }
  if (model.rope_freq_scale.has_value()) {
    rope_settings.emplace("freqScale", JsonValue(model.rope_freq_scale.value()));
  }

  JsonObject attention;
  if (model.attention_head_count.has_value()) {
    attention.emplace("headCount", JsonValue(model.attention_head_count.value()));
  }
  if (model.kv_head_count.has_value()) {
    attention.emplace("kvHeadCount", JsonValue(model.kv_head_count.value()));
  }
  if (model.layer_norm_rms_epsilon.has_value()) {
    attention.emplace("layerNormRmsEpsilon", JsonValue(model.layer_norm_rms_epsilon.value()));
  }

  JsonObject output = {
    {"format", JsonValue("gguf")},
    {"filePath", JsonValue(model.file_path.string())},
    {"fileSizeBytes", JsonValue(static_cast<std::uint64_t>(model.file_size_bytes))},
    {"memoryMode", JsonValue(model.memory_mode)},
    {"pinnedMemory", JsonValue(model.pinned_memory)},
    {"version", JsonValue(static_cast<std::uint64_t>(model.version))},
    {"metadataHash", JsonValue(model.metadata_hash)},
    {"architecture", JsonValue(model.architecture.empty() ? "unknown" : model.architecture)},
    {"quantization", JsonValue(model.quantization.empty() ? "unknown" : model.quantization)},
    {"tokenizer", JsonValue(model.tokenizer.empty() ? "unknown" : model.tokenizer)},
    {"chatTemplatePresent", JsonValue(!model.chat_template.empty())},
    {"tensorCount", JsonValue(model.tensor_count)},
    {"tensorDirectoryLoaded", JsonValue(true)},
    {"tensorDataOffsetBytes", JsonValue(model.tensor_data_offset)},
    {"tensorDataBytes", JsonValue(model.tensor_data_bytes)},
    {"tensorsPreview", build_tensors_preview(model)},
    {"specialTokens", JsonValue(model.special_tokens)},
    {"specialTokenTexts", JsonValue(model.special_token_texts)},
    {"tokenVocabularySize", JsonValue(model.token_vocabulary_size)},
    {"tokenPreview", JsonValue(model.token_preview)},
    {"tokenMergeCount", JsonValue(model.token_merge_count)},
    {"mergePreview", JsonValue(model.merge_preview)},
    {"warnings", JsonValue(JsonArray {})},
  };

  JsonArray warnings;
  for (const std::string &warning : model.warnings) {
    warnings.push_back(JsonValue(warning));
  }
  output["warnings"] = JsonValue(warnings);

  if (model.context_length.has_value()) {
    output.emplace("contextLength", JsonValue(model.context_length.value()));
  }
  if (model.embedding_length.has_value()) {
    output.emplace("embeddingLength", JsonValue(model.embedding_length.value()));
  }
  if (model.block_count.has_value()) {
    output.emplace("blockCount", JsonValue(model.block_count.value()));
  }
  if (!attention.empty()) {
    output.emplace("attention", JsonValue(attention));
  }
  if (!rope_settings.empty()) {
    output.emplace("rope", JsonValue(rope_settings));
  }
  if (model.parameter_estimate_b.has_value()) {
    output.emplace("parameterEstimateB", JsonValue(model.parameter_estimate_b.value()));
  }
  if (!model.executor_shape.empty()) {
    output.emplace("executorShape", JsonValue(model.executor_shape));
  }
  return output;
}

// --- KV cache storage modes -------------------------------------------------
// LIZARD_NATIVE_KV_CACHE selects how the layered KV cache stores entries:
//   f32  (default) - float32, byte-identical to the historical behaviour.
//   f16            - float16 storage, dequantized to f32 on read. Halves KV
//                    memory. The GPU lanes already round KV through f16 on
//                    upload (staging_float_to_half), so f16 storage matches
//                    the precision the resident GPU path always had.
//   q8_0           - symmetric 8-bit blocks of 32 with an f32 scale
//                    (9 bits/element measured, ~3.6x vs f32).
//   q4_0           - symmetric 4-bit blocks of 32 with an f32 scale, two
//                    nibbles per byte (5 bits/element, ~6.4x vs f32). The
//                    long-context option; expect small quality drift.
//   q3_wht         - Walsh-Hadamard-rotated 3-bit blocks of 32 (12 bytes of
//                    quants + f32 scale = 4 bits/element, 8x vs f32).
//
// q3_wht implements the rotation core of TurboQuant (arXiv 2504.19874).
// Straight 3-bit quantization of KV blocks is poor because attention keys
// carry outlier channels: one large component forces a large scale and the
// other 31 values collapse onto a couple of grid levels. A Walsh-Hadamard
// transform is an orthogonal rotation that smears any outlier across all 32
// coordinates, so the rotated block is near-Gaussian and a uniform 3-bit
// grid fits it much better. The transform needs no training or calibration
// (TurboQuant's "data-oblivious" property), costs only adds/subs, and
// H*H = 32*I makes the inverse the same butterfly followed by a 1/32 scale.
// The paper's 1-bit QJL residual corrector is NOT implemented; this is the
// rotation half only (see enhancement_phases.md 3.1b).
enum class KvCacheMode : std::uint8_t { F32 = 0, F16 = 1, Q8_0 = 2, Q4_0 = 3, Q3_WHT = 4 };

// In-place unnormalized Walsh-Hadamard butterfly over a power-of-two count.
static void walsh_hadamard_in_place(float *values, std::size_t count) {
  for (std::size_t stride = 1; stride < count; stride <<= 1) {
    for (std::size_t base = 0; base < count; base += stride << 1) {
      for (std::size_t item = base; item < base + stride; item += 1) {
        const float a = values[item];
        const float b = values[item + stride];
        values[item] = a + b;
        values[item + stride] = a - b;
      }
    }
  }
}

static KvCacheMode kv_cache_mode() {
  static const KvCacheMode mode = [] {
    const char *value = std::getenv("LIZARD_NATIVE_KV_CACHE");
    if (value == nullptr) {
      // EnhancementRun_04 Phase 1. f16 halves the KV cache -- 224 KiB per token
      // down to 112 on llama-3.2-3b, so a 4096-token sequence costs 458 MB
      // instead of 917 -- with a byte-identical greedy token stream and no
      // measurable decode cost (7.27/7.30 vs f32's 7.22/7.26 across two quiet
      // runs). On a 16 GB UMA box that halving is what a larger decode batch
      // is paid for out of.
      //
      // NOT q8_0, though it is 3.76x smaller and equally byte-identical: it
      // costs ~6% decode (6.79/6.87), and the phase gate requires a default to
      // be free. An earlier measurement on a loaded machine put q8_0 ahead of
      // f16 and would have picked it; the quiet re-run reversed the order,
      // which is the whole reason that gate exists. q8_0 stays one env var away
      // for anyone trading throughput for context length.
      return KvCacheMode::F16;
    }
    std::string text(value);
    for (char &character : text) {
      character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    if (text == "f16" || text == "fp16" || text == "half") {
      return KvCacheMode::F16;
    }
    if (text == "q8_0" || text == "q8" || text == "int8") {
      return KvCacheMode::Q8_0;
    }
    if (text == "q4_0" || text == "q4") {
      return KvCacheMode::Q4_0;
    }
    if (text == "q3_wht" || text == "q3" || text == "turbo3") {
      return KvCacheMode::Q3_WHT;
    }
    return KvCacheMode::F32;
  }();
  return mode;
}

static const char *kv_cache_mode_name() {
  switch (kv_cache_mode()) {
    case KvCacheMode::F16:
      return "f16";
    case KvCacheMode::Q8_0:
      return "q8_0";
    case KvCacheMode::Q4_0:
      return "q4_0";
    case KvCacheMode::Q3_WHT:
      return "q3_wht";
    default:
      return "f32";
  }
}

// True for lanes with no device-side upload path: the Gemma GPU lanes must
// fall back to CPU attention for these.
static bool kv_cache_mode_is_quantized() {
  const KvCacheMode mode = kv_cache_mode();
  return mode == KvCacheMode::Q8_0 || mode == KvCacheMode::Q4_0 || mode == KvCacheMode::Q3_WHT;
}

static std::uint16_t float_to_half(float value) {
  std::uint32_t f32_bits = 0;
  std::memcpy(&f32_bits, &value, sizeof(f32_bits));
  const std::uint16_t sign = static_cast<std::uint16_t>((f32_bits >> 16) & 0x8000U);
  const std::uint32_t exponent = (f32_bits >> 23) & 0xFFU;
  const std::uint32_t mantissa = f32_bits & 0x007FFFFFU;

  if (exponent == 0xFFU) {
    // Inf / NaN: keep NaN payload bit so NaN stays NaN after truncation.
    return static_cast<std::uint16_t>(sign | 0x7C00U | (mantissa != 0 ? 0x0200U : 0));
  }
  const std::int32_t unbiased = static_cast<std::int32_t>(exponent) - 127;
  if (unbiased > 15) {
    return static_cast<std::uint16_t>(sign | 0x7C00U); // overflow -> inf
  }
  if (unbiased >= -14) {
    // Normal half. Round to nearest even on the 13 dropped mantissa bits.
    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1FFFU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U) != 0)) {
      half_mantissa += 1;
      if (half_mantissa == 0x400U) {
        half_mantissa = 0;
        if (unbiased + 1 > 15) {
          return static_cast<std::uint16_t>(sign | 0x7C00U);
        }
        return static_cast<std::uint16_t>(sign | ((unbiased + 1 + 15) << 10));
      }
    }
    return static_cast<std::uint16_t>(sign | ((unbiased + 15) << 10) | half_mantissa);
  }
  if (unbiased >= -24) {
    // Subnormal half.
    const std::uint32_t full_mantissa = mantissa | 0x00800000U;
    const std::uint32_t shift = static_cast<std::uint32_t>(-unbiased - 14 + 13);
    std::uint32_t half_mantissa = full_mantissa >> shift;
    const std::uint32_t remainder = full_mantissa & ((1U << shift) - 1);
    const std::uint32_t halfway = 1U << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U) != 0)) {
      half_mantissa += 1;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }
  return sign; // underflow -> signed zero
}

// --- Attention microkernels (enhancement Phase 2.1a) ------------------------
// The decode attention inner loops are dot(query, key) and
// context += weight * value over head_dim (64..256) elements. On the AVX2
// build these run 8-wide with FMA; LIZARD_NATIVE_ATTN_SIMD=0 restores the
// scalar loops (byte-identical to the historical summation order, useful when
// bisecting numeric drift).
static bool attention_simd_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_ATTN_SIMD");
    return value == nullptr || std::string(value) != "0";
  }();
  return enabled;
}

static float attention_dot(const float *a, const float *b, std::size_t count) {
#ifdef LIZARD_NATIVE_AVX2
  if (attention_simd_enabled() && count >= 8) {
    __m256 accumulator0 = _mm256_setzero_ps();
    __m256 accumulator1 = _mm256_setzero_ps();
    std::size_t index = 0;
    for (; index + 16 <= count; index += 16) {
      accumulator0 = _mm256_fmadd_ps(
        _mm256_loadu_ps(a + index), _mm256_loadu_ps(b + index), accumulator0);
      accumulator1 = _mm256_fmadd_ps(
        _mm256_loadu_ps(a + index + 8), _mm256_loadu_ps(b + index + 8), accumulator1);
    }
    for (; index + 8 <= count; index += 8) {
      accumulator0 = _mm256_fmadd_ps(
        _mm256_loadu_ps(a + index), _mm256_loadu_ps(b + index), accumulator0);
    }
    const __m256 combined = _mm256_add_ps(accumulator0, accumulator1);
    const __m128 low = _mm256_castps256_ps128(combined);
    const __m128 high = _mm256_extractf128_ps(combined, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float total = _mm_cvtss_f32(sum);
    for (; index < count; index += 1) {
      total += a[index] * b[index];
    }
    return total;
  }
#endif
  float total = 0.0f;
  for (std::size_t index = 0; index < count; index += 1) {
    total += a[index] * b[index];
  }
  return total;
}

static void attention_axpy(float *destination, const float *source, float weight, std::size_t count) {
#ifdef LIZARD_NATIVE_AVX2
  if (attention_simd_enabled() && count >= 8) {
    const __m256 weight_vector = _mm256_set1_ps(weight);
    std::size_t index = 0;
    for (; index + 8 <= count; index += 8) {
      _mm256_storeu_ps(
        destination + index,
        _mm256_fmadd_ps(
          weight_vector,
          _mm256_loadu_ps(source + index),
          _mm256_loadu_ps(destination + index)));
    }
    for (; index < count; index += 1) {
      destination[index] += weight * source[index];
    }
    return;
  }
#endif
  for (std::size_t index = 0; index < count; index += 1) {
    destination[index] += weight * source[index];
  }
}

// One K or V lane of a layer's KV cache. Storage format is fixed for the
// process lifetime (env-driven), so mode branches are perfectly predictable.
// f32 mode hands out direct pointers (zero-copy, byte-identical to the old
// std::vector<float> behaviour); f16/q8_0 dequantize rows into the caller's
// scratch buffer.
class KvLane {
 public:
  KvLane() : mode_(kv_cache_mode()) {}

  std::size_t size() const {
    return logical_;
  }

  void append(const float *source, std::size_t count) {
    logical_ += count;
    switch (mode_) {
      case KvCacheMode::F32:
        f32_.insert(f32_.end(), source, source + count);
        return;
      case KvCacheMode::F16: {
        const std::size_t base = f16_.size();
        f16_.resize(base + count);
        for (std::size_t item = 0; item < count; item += 1) {
          f16_[base + item] = float_to_half(source[item]);
        }
        return;
      }
      case KvCacheMode::Q8_0:
      case KvCacheMode::Q4_0:
      case KvCacheMode::Q3_WHT: {
        q8_pending_.insert(q8_pending_.end(), source, source + count);
        while (q8_pending_.size() >= k_q8_block) {
          quantize_pending_block();
        }
        return;
      }
    }
  }

  // Returns `count` contiguous f32 elements starting at `offset`. For f32
  // mode this is a direct pointer into storage; otherwise the row is
  // dequantized into `scratch` (resized as needed) and scratch's data is
  // returned. `scratch` must stay alive while the pointer is used.
  const float *row_f32(std::size_t offset, std::size_t count, std::vector<float> &scratch) const {
    if (mode_ == KvCacheMode::F32) {
      return f32_.data() + offset;
    }
    if (scratch.size() < count) {
      scratch.resize(count);
    }
    if (mode_ == KvCacheMode::F16) {
      std::size_t item = 0;
#ifdef LIZARD_NATIVE_AVX2
      // F16C converts 8 halves per instruction; the AVX2 build variant
      // compiles with -mf16c (all AVX2-capable CPUs implement F16C).
      for (; item + 8 <= count; item += 8) {
        _mm256_storeu_ps(
          scratch.data() + item,
          _mm256_cvtph_ps(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(f16_.data() + offset + item))));
      }
#endif
      for (; item < count; item += 1) {
        scratch[item] = half_to_float(f16_[offset + item]);
      }
      return scratch.data();
    }
    const std::size_t quantized_elements = q8_scales_.size() * k_q8_block;
    if (mode_ == KvCacheMode::Q3_WHT) {
      // Rotated blocks reconstruct 32 elements at a time: walk the covering
      // blocks, inverse-transform each into a local buffer, and copy the
      // requested slice. The pending tail is raw f32 as in the other modes.
      float block_values[k_q8_block];
      std::size_t item = 0;
      while (item < count) {
        const std::size_t index = offset + item;
        if (index >= quantized_elements) {
          scratch[item] = q8_pending_[index - quantized_elements];
          item += 1;
          continue;
        }
        const std::size_t block = index / k_q8_block;
        dequant_q3_block(block, block_values);
        const std::size_t begin = index % k_q8_block;
        const std::size_t take = std::min(k_q8_block - begin, count - item);
        for (std::size_t inner = 0; inner < take; inner += 1) {
          scratch[item + inner] = block_values[begin + inner];
        }
        item += take;
      }
      return scratch.data();
    }
    for (std::size_t item = 0; item < count; item += 1) {
      const std::size_t index = offset + item;
      if (index >= quantized_elements) {
        scratch[item] = q8_pending_[index - quantized_elements];
      } else if (mode_ == KvCacheMode::Q8_0) {
        scratch[item] = static_cast<float>(q8_quants_[index]) * q8_scales_[index / k_q8_block];
      } else {
        scratch[item] = static_cast<float>(unpack_nibble(index)) * q8_scales_[index / k_q8_block];
      }
    }
    return scratch.data();
  }

  // The GPU upload paths require the historical contiguous f32 vector. Only
  // callable in f32 mode; every GPU lane is gated on kv_cache_mode() == F32.
  const std::vector<float> &f32_vector() const {
    return f32_;
  }

  // Seed view for the device KV cache. The device cache stores IEEE halves,
  // so an f16 lane uploads with a straight copy and an f32 lane converts in
  // the packing loop. q8_0 has no direct upload path yet: the view comes back
  // unpopulated and the caller must keep that layer on the CPU lanes.
  D3d12KvLaneView gpu_lane_view() const {
    D3d12KvLaneView view;
    view.count = logical_;
    if (mode_ == KvCacheMode::F32) {
      view.f32 = f32_.data();
      view.populated = true;
    } else if (mode_ == KvCacheMode::F16) {
      view.f16 = f16_.data();
      view.populated = true;
    }
    return view;
  }

  std::uint64_t storage_bytes() const {
    switch (mode_) {
      case KvCacheMode::F32:
        return static_cast<std::uint64_t>(f32_.size() * sizeof(float));
      case KvCacheMode::F16:
        return static_cast<std::uint64_t>(f16_.size() * sizeof(std::uint16_t));
      case KvCacheMode::Q8_0:
        return static_cast<std::uint64_t>(
          q8_quants_.size() * sizeof(std::int8_t)
          + q8_scales_.size() * sizeof(float)
          + q8_pending_.size() * sizeof(float));
      case KvCacheMode::Q4_0:
        return static_cast<std::uint64_t>(
          q4_nibbles_.size() * sizeof(std::uint8_t)
          + q8_scales_.size() * sizeof(float)
          + q8_pending_.size() * sizeof(float));
      case KvCacheMode::Q3_WHT:
        return static_cast<std::uint64_t>(
          q3_bits_.size() * sizeof(std::uint8_t)
          + q8_scales_.size() * sizeof(float)
          + q8_pending_.size() * sizeof(float));
    }
    return 0;
  }

  // Drops everything past the first `elements` entries. Used by prefix reuse
  // when a new prompt diverges from the cached sequence mid-way. For q8_0 the
  // boundary block's kept values are dequantized back into the pending tail,
  // so they get re-quantized together with the next appended tokens; the
  // extra rounding is confined to that single 32-element block.
  void truncate(std::size_t elements) {
    if (elements >= logical_) {
      return;
    }
    logical_ = elements;
    switch (mode_) {
      case KvCacheMode::F32:
        f32_.resize(elements);
        return;
      case KvCacheMode::F16:
        f16_.resize(elements);
        return;
      case KvCacheMode::Q8_0:
      case KvCacheMode::Q4_0:
      case KvCacheMode::Q3_WHT: {
        const std::size_t full_blocks = elements / k_q8_block;
        const std::size_t remainder = elements % k_q8_block;
        std::vector<float> tail(remainder);
        const std::size_t quantized_elements = q8_scales_.size() * k_q8_block;
        float block_values[k_q8_block];
        std::size_t decoded_block = static_cast<std::size_t>(-1);
        for (std::size_t item = 0; item < remainder; item += 1) {
          const std::size_t index = full_blocks * k_q8_block + item;
          if (index >= quantized_elements) {
            tail[item] = q8_pending_[index - quantized_elements];
          } else if (mode_ == KvCacheMode::Q8_0) {
            tail[item] = static_cast<float>(q8_quants_[index]) * q8_scales_[index / k_q8_block];
          } else if (mode_ == KvCacheMode::Q4_0) {
            tail[item] = static_cast<float>(unpack_nibble(index)) * q8_scales_[index / k_q8_block];
          } else {
            const std::size_t block = index / k_q8_block;
            if (block != decoded_block) {
              dequant_q3_block(block, block_values);
              decoded_block = block;
            }
            tail[item] = block_values[index % k_q8_block];
          }
        }
        if (q8_scales_.size() > full_blocks) {
          q8_scales_.resize(full_blocks);
          if (mode_ == KvCacheMode::Q8_0) {
            q8_quants_.resize(full_blocks * k_q8_block);
          } else if (mode_ == KvCacheMode::Q4_0) {
            q4_nibbles_.resize(full_blocks * (k_q8_block / 2));
          } else {
            q3_bits_.resize(full_blocks * k_q3_block_bytes);
          }
        }
        q8_pending_ = std::move(tail);
        return;
      }
    }
  }

 private:
  static constexpr std::size_t k_q8_block = 32;
  static constexpr std::size_t k_q3_block_bytes = 12; // 32 elements x 3 bits

  // Signed 4-bit value at a flat element index (q4_0 lanes only). Two nibbles
  // per byte, low nibble first; stored biased by +8 so the nibble is unsigned.
  std::int32_t unpack_nibble(std::size_t index) const {
    const std::uint8_t byte = q4_nibbles_[index / 2];
    const std::uint8_t nibble = (index % 2 == 0)
      ? static_cast<std::uint8_t>(byte & 0x0FU)
      : static_cast<std::uint8_t>(byte >> 4);
    return static_cast<std::int32_t>(nibble) - 8;
  }

  // 3-bit codes are packed little-endian across the 12-byte block payload:
  // element i occupies bits [3i, 3i+3), possibly spanning two bytes. Stored
  // biased by +4 so codes are unsigned ([-4,3] -> [0,7]).
  void pack_q3_code(std::size_t block, std::size_t item, std::uint32_t code) {
    const std::size_t bit = item * 3;
    const std::size_t base = block * k_q3_block_bytes + bit / 8;
    const std::uint32_t shift = static_cast<std::uint32_t>(bit % 8);
    q3_bits_[base] = static_cast<std::uint8_t>(q3_bits_[base] | ((code << shift) & 0xFFU));
    if (shift > 5) {
      q3_bits_[base + 1] = static_cast<std::uint8_t>(q3_bits_[base + 1] | (code >> (8 - shift)));
    }
  }

  std::int32_t unpack_q3_code(std::size_t block, std::size_t item) const {
    const std::size_t bit = item * 3;
    const std::size_t base = block * k_q3_block_bytes + bit / 8;
    const std::uint32_t shift = static_cast<std::uint32_t>(bit % 8);
    std::uint32_t raw = q3_bits_[base];
    if (shift > 5) {
      raw |= static_cast<std::uint32_t>(q3_bits_[base + 1]) << 8;
    }
    return static_cast<std::int32_t>((raw >> shift) & 0x7U) - 4;
  }

  // Reconstructs one rotated 3-bit block: codes -> rotated values -> inverse
  // Walsh-Hadamard (same butterfly, then 1/32 since H*H = 32*I).
  void dequant_q3_block(std::size_t block, float *out) const {
    const float scale = q8_scales_[block];
    for (std::size_t item = 0; item < k_q8_block; item += 1) {
      out[item] = static_cast<float>(unpack_q3_code(block, item)) * scale;
    }
    walsh_hadamard_in_place(out, k_q8_block);
    for (std::size_t item = 0; item < k_q8_block; item += 1) {
      out[item] *= (1.0f / static_cast<float>(k_q8_block));
    }
  }

  void quantize_pending_block() {
    if (mode_ == KvCacheMode::Q3_WHT) {
      // Rotate first, then quantize the rotated coordinates. The scale is
      // amax/3.5 with codes clamped to [-4,3]: the grid's asymmetric extra
      // negative level is used and only the single most-positive rotated
      // coordinate can clip by half a step.
      float rotated[k_q8_block];
      for (std::size_t item = 0; item < k_q8_block; item += 1) {
        rotated[item] = q8_pending_[item];
      }
      walsh_hadamard_in_place(rotated, k_q8_block);
      float amax = 0.0f;
      for (std::size_t item = 0; item < k_q8_block; item += 1) {
        amax = std::max(amax, std::abs(rotated[item]));
      }
      const float scale = amax / 3.5f;
      const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
      const std::size_t block = q8_scales_.size();
      q8_scales_.push_back(scale);
      q3_bits_.resize(q3_bits_.size() + k_q3_block_bytes, 0);
      for (std::size_t item = 0; item < k_q8_block; item += 1) {
        const float scaled = rotated[item] * inverse;
        const float rounded = scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f;
        const std::int32_t clamped = static_cast<std::int32_t>(
          std::max(-4.0f, std::min(3.0f, rounded)));
        pack_q3_code(block, item, static_cast<std::uint32_t>(clamped + 4));
      }
      q8_pending_.erase(q8_pending_.begin(), q8_pending_.begin() + static_cast<std::ptrdiff_t>(k_q8_block));
      return;
    }

    float amax = 0.0f;
    for (std::size_t item = 0; item < k_q8_block; item += 1) {
      amax = std::max(amax, std::abs(q8_pending_[item]));
    }
    // Symmetric max-scale: q8_0 spans [-127,127], q4_0 spans [-8,7] and is
    // scaled by 7 so the positive side saturates rather than the negative.
    const float range = mode_ == KvCacheMode::Q8_0 ? 127.0f : 7.0f;
    const float scale = amax / range;
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    q8_scales_.push_back(scale);

    if (mode_ == KvCacheMode::Q8_0) {
      const std::size_t base = q8_quants_.size();
      q8_quants_.resize(base + k_q8_block);
      for (std::size_t item = 0; item < k_q8_block; item += 1) {
        const float scaled = q8_pending_[item] * inverse;
        const float rounded = scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f;
        q8_quants_[base + item] = static_cast<std::int8_t>(
          std::max(-127.0f, std::min(127.0f, rounded)));
      }
    } else {
      const std::size_t base = q4_nibbles_.size();
      q4_nibbles_.resize(base + k_q8_block / 2, 0);
      for (std::size_t item = 0; item < k_q8_block; item += 1) {
        const float scaled = q8_pending_[item] * inverse;
        const float rounded = scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f;
        const std::int32_t clamped = static_cast<std::int32_t>(
          std::max(-8.0f, std::min(7.0f, rounded)));
        const std::uint8_t nibble = static_cast<std::uint8_t>((clamped + 8) & 0x0F);
        std::uint8_t &byte = q4_nibbles_[base + item / 2];
        if (item % 2 == 0) {
          byte = static_cast<std::uint8_t>((byte & 0xF0U) | nibble);
        } else {
          byte = static_cast<std::uint8_t>((byte & 0x0FU) | (nibble << 4));
        }
      }
    }
    q8_pending_.erase(q8_pending_.begin(), q8_pending_.begin() + static_cast<std::ptrdiff_t>(k_q8_block));
  }

  KvCacheMode mode_;
  std::size_t logical_ = 0;
  std::vector<float> f32_;
  std::vector<std::uint16_t> f16_;
  std::vector<std::int8_t> q8_quants_;
  std::vector<std::uint8_t> q4_nibbles_;
  std::vector<std::uint8_t> q3_bits_;
  std::vector<float> q8_scales_;
  std::vector<float> q8_pending_;
};

struct SessionState {
  struct LayerKvCache {
    KvLane keys;
    KvLane values;
  };
  struct PrefillProjectionMetric {
    std::string label;
    std::int64_t layer_index = -1;
    std::uint64_t batch_count = 0;
    std::uint64_t row_count = 0;
    std::uint64_t row_width = 0;
    std::string backend = "cpu";
    std::uint64_t elapsed_ms = 0;
  };
  struct DecodePhaseMetric {
    std::string label;
    std::int64_t layer_index = -1;
    std::string backend = "cpu";
    std::uint64_t elapsed_ms = 0;
  };
  struct SamplingState {
    double temperature = 0.0;
    std::uint64_t top_k = 0;
    double top_p = 1.0;
    std::uint64_t seed = 0;
    std::mt19937_64 rng = std::mt19937_64(0);
    bool initialized = false;
  };
  std::string session_id;
  // Sessions cloned from the same completed prefill carry the same opaque
  // generation. Greedy decode may coalesce them only while their exact token
  // history still matches; a fresh/re-prefilled session receives a new value.
  std::uint64_t decode_equivalence_id = 0;
  bool cancelled = false;
  std::string prompt;
  std::vector<std::uint64_t> prompt_token_ids;
  std::vector<std::uint64_t> sequence_token_ids;
  std::string prompt_token_mode;
  std::uint64_t kv_cache_bytes = 0;
  std::uint64_t max_tokens = 0;
  // Extra stop strings supplied per generate call (chat end-of-turn markers,
  // instruction-format headers). The decode loop already stops on the GGUF EOS
  // and "<turn|>"; these cover instruct families whose turn-ender is a normal
  // token (<|eot_id|>, <|im_end|>, ...) and base models that never emit EOS at
  // all. Matched against the decoded text tail so a marker split across tokens
  // is still caught.
  std::vector<std::string> stop_sequences;
  // EnhancementRun_04 Phase 4: the batched decode path steps a session one
  // token per call, so the rolling stop-marker tail cannot live on the stack of
  // a decode loop the way the single-sequence path's does. It belongs to the
  // session because that is what persists between steps.
  std::string batch_stop_scan_tail;
  std::vector<LayerKvCache> layered_kv_cache;
  std::uint64_t gpu_kv_cache_generation = 0;
  std::uint64_t gpu_kv_max_positions = 0;
  std::vector<bool> gpu_kv_cache_active;
  // Session-cached whole-token-graph descriptors: layer refs, PLE config
  // (minus the per-token inputs pointer), and the LM-head ref are invariant
  // per KV-cache generation; only attention.position and the PLE token row
  // change per token. Invalidated by reset_layered_session_state.
  std::vector<D3d12GemmaTokenLayer> gpu_token_graph_layers;
  D3d12GemmaPleInputConfig gpu_token_graph_ple;
  D3d12ResidentWeightRef gpu_token_graph_output_ref;
  bool gpu_token_graph_descriptors_ready = false;
  std::uint64_t gpu_token_graph_descriptor_generation = 0;
  std::vector<float> last_hidden_state;
  std::vector<float> last_logits;
  // GPU-selected greedy token for the last consumed position when the token
  // graph skipped the logits readback; -1 when logits are authoritative.
  std::int64_t last_gpu_argmax_token = -1;
  std::uint64_t cached_token_count = 0;
  std::uint64_t generated_token_count = 0;
  std::uint64_t last_decode_token_count = 0;
  std::uint64_t last_prefill_ms = 0;
  std::uint64_t last_decode_ms = 0;
  // Cheap, in-process evidence for the most recent decode window. These
  // counters replace the external PowerShell sampler for normal command,
  // chat, and serve traffic. They are reset at prefill and accumulated across
  // decode_batch steps, so both single-shot and cohort execution report the
  // same contract.
  std::uint64_t resource_decode_elapsed_ms = 0;
  std::uint64_t resource_decode_cpu_100ns = 0;
  std::uint64_t resource_decode_token_count = 0;
  std::uint64_t resource_working_set_bytes = 0;
  std::uint64_t resource_private_bytes = 0;
  std::uint64_t resource_peak_working_set_bytes = 0;
  std::uint64_t resource_peak_private_bytes = 0;
  bool resource_metrics_available = false;
  bool resource_metrics_cohort = false;
  bool prefill_complete = false;
  std::string native_backend = ggml_cpu_backend_effective_name();
  std::string native_backend_detail;
  std::string prefill_backend = ggml_cpu_backend_effective_name();
  std::string prefill_backend_detail;
  std::string decode_backend = ggml_cpu_backend_effective_name();
  std::string decode_backend_detail;
  std::string gpu_decode_policy = "allowed";
  bool decode_qkv_fuse_blocked = false;
  std::string decode_qkv_fuse_detail;
  bool decode_ffn_fuse_blocked = false;
  std::string decode_ffn_fuse_detail;
  bool decode_layer_tail_fuse_blocked = false;
  std::string decode_layer_tail_fuse_detail;
  std::vector<PrefillProjectionMetric> prefill_projection_metrics;
  std::vector<DecodePhaseMetric> decode_phase_metrics;
  SamplingState sampling;
  // When true, generate emits a "generate.token" JSON-RPC notification per
  // decoded token so servers can stream deltas while decoding is in flight.
  bool stream_tokens = false;
};

static std::uint64_t estimate_session_native_bytes(const SessionState &session);
static JsonObject build_session_metrics_object(const SessionState &session);
static JsonValue build_cpu_backend_object();
static JsonValue build_gpu_adapter_object();
static JsonValue build_gpu_memory_object();

static JsonArray build_prefill_projection_metrics_array(const SessionState &session) {
  JsonArray metrics;
  metrics.reserve(session.prefill_projection_metrics.size());
  for (const SessionState::PrefillProjectionMetric &metric : session.prefill_projection_metrics) {
    JsonObject metric_object = {
      {"label", JsonValue(metric.label)},
      {"batchCount", JsonValue(metric.batch_count)},
      {"rowCount", JsonValue(metric.row_count)},
      {"rowWidth", JsonValue(metric.row_width)},
      {"backend", JsonValue(metric.backend.empty() ? "cpu" : metric.backend)},
      {"elapsedMs", JsonValue(metric.elapsed_ms)},
    };
    if (metric.layer_index >= 0) {
      metric_object.emplace("layerIndex", JsonValue(static_cast<std::uint64_t>(metric.layer_index)));
    }
    metrics.push_back(JsonValue(metric_object));
  }
  return metrics;
}

static JsonArray build_decode_phase_metrics_array(const SessionState &session) {
  JsonArray metrics;
  metrics.reserve(session.decode_phase_metrics.size());
  for (const SessionState::DecodePhaseMetric &metric : session.decode_phase_metrics) {
    JsonObject metric_object = {
      {"label", JsonValue(metric.label)},
      {"backend", JsonValue(metric.backend.empty() ? "cpu" : metric.backend)},
      {"elapsedMs", JsonValue(metric.elapsed_ms)},
    };
    if (metric.layer_index >= 0) {
      metric_object.emplace("layerIndex", JsonValue(static_cast<std::uint64_t>(metric.layer_index)));
    }
    metrics.push_back(JsonValue(metric_object));
  }
  return metrics;
}

struct MatrixShape {
  std::uint64_t row_width = 0;
  std::uint64_t row_count = 0;
};

struct BatchExecutionConfig {
  bool allow_hybrid_gpu = false;
  bool hybrid_gpu_used = false;
  std::uint64_t gpu_tile_rows = 512;
  std::uint64_t min_gpu_prefill_tokens = 48;
  std::string native_backend = ggml_cpu_backend_effective_name();
  std::string detail;
  const void *cached_gpu_input_owner = nullptr;
  std::vector<float> cached_gpu_packed_inputs;
  std::uint32_t cached_gpu_batch_count = 0;
  std::uint32_t cached_gpu_input_width = 0;
  std::int64_t current_layer_index = -1;
  std::vector<SessionState::PrefillProjectionMetric> *prefill_projection_metrics = nullptr;
};

struct GemmaLayerExecutionConfig {
  std::uint64_t hidden_size = 0;
  std::uint64_t head_count = 0;
  std::uint64_t kv_head_count = 0;
  std::uint64_t head_dim = 0;
  std::uint64_t query_size = 0;
  std::uint64_t kv_size = 0;
  std::uint64_t attention_window = 0;
  double rope_freq_base = 10000.0;
  double rope_freq_scale = 1.0;
  double rms_norm_epsilon = 1e-5;
  std::vector<float> rope_freq_factors;
  MatrixShape query_shape;
  MatrixShape key_shape;
  MatrixShape value_shape;
  MatrixShape output_shape;
  MatrixShape gate_shape;
  MatrixShape down_shape;
  MatrixShape up_shape;
  std::uint64_t per_layer_input_size = 0;
  MatrixShape per_layer_input_gate_shape;
  MatrixShape per_layer_projection_shape;
};

struct LlamaExecutionConfig {
  std::uint64_t hidden_size = 0;
  std::uint64_t head_count = 0;
  std::uint64_t kv_head_count = 0;
  std::uint64_t head_dim = 0;
  double rope_freq_base = 10000.0;
  double rope_freq_scale = 1.0;
  std::vector<float> rope_freq_factors;
};

// Load-time cache of the seven per-layer projection shapes on the plain
// llama decode path. Gemma got the equivalent cache in the 2026-07-02 pass
// (GemmaLayerExecutionConfig); the llama path recomputed every shape per
// projection per token until Phase 3 iteration 1.
struct LlamaLayerShapeConfig {
  MatrixShape query_shape;
  MatrixShape key_shape;
  MatrixShape value_shape;
  MatrixShape output_shape;
  MatrixShape gate_shape;
  MatrixShape down_shape;
  MatrixShape up_shape;
};

struct LoadedModelState {
  ParsedGgufModel model;
  JsonObject compatibility;
  LlamaTensorIndex tensor_index;
  // Empty for a text-only model; `ready` is what gates any vision work.
  VisionTensorIndex vision_index;
  // A separately-loaded mmproj, for the many families that ship the vision
  // tower as its own GGUF (LLaVA, gemma-3, most of the Hub) rather than packed
  // beside the language model the way Gemma 3n does. It is a full model state
  // of its own so every existing facility - mmap, dequantization, the tensor
  // cache - applies to it unchanged. shared_ptr rather than unique_ptr because
  // the type is still incomplete here.
  std::shared_ptr<LoadedModelState> projector_model;
  mutable std::map<std::size_t, std::vector<float>> vector_tensor_cache;
  std::vector<GemmaLayerExecutionConfig> gemma_layer_configs;
  bool gemma_layer_configs_ready = false;
  std::vector<LlamaLayerShapeConfig> llama_layer_shapes;
  bool llama_layer_shapes_ready = false;
  mutable std::optional<LlamaExecutionConfig> llama_execution_config_cache;
  mutable bool llama_execution_config_resolved = false;
  bool gpu_projection_candidate = false;
  std::vector<std::size_t> sorted_token_indices;
  std::map<std::string, std::size_t> token_to_id;
  std::map<std::string, std::size_t> merge_ranks;
  std::unique_ptr<FileView> file_view;
  std::uint64_t load_ms = 0;
};

struct WorkerState {
  std::uint32_t rpc_version = 1;
  std::unique_ptr<LoadedModelState> loaded_model;
  std::uint64_t load_ms = 0;
  std::map<std::string, SessionState> sessions;
  JsonValue last_metrics;
  bool shutdown_requested = false;
};

static const TensorInfo &tensor_at(const LoadedModelState &state, std::size_t index) {
  if (index == k_missing_tensor_index || index >= state.model.tensors.size()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Requested tensor index is unavailable in the native model state.");
  }
  return state.model.tensors[index];
}

static const std::uint8_t *tensor_data_ptr(const LoadedModelState &state, const TensorInfo &tensor) {
  if (tensor.absolute_offset > state.model.file_size_bytes) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Tensor offset extends beyond the mapped GGUF file.");
  }
  if (tensor.byte_length > 0 && tensor.absolute_offset + tensor.byte_length > state.model.file_size_bytes) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Tensor payload extends beyond the mapped GGUF file.");
  }
  return state.file_view->data() + tensor.absolute_offset;
}

static void dequantize_tensor_range_into_f32(
  const LoadedModelState &state,
  const TensorInfo &tensor,
  std::uint64_t start,
  std::uint64_t count,
  std::vector<float> &values) {
  if (start > tensor.element_count || count > tensor.element_count - start) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Requested tensor range is out of bounds.");
  }
  values.assign(static_cast<std::size_t>(count), 0.0f);
  if (count == 0) {
    return;
  }

  const std::uint64_t end = start + count;
  const std::uint8_t *data = tensor_data_ptr(state, tensor);
  const auto store_value = [&](std::uint64_t flat_index, float value) {
    if (flat_index >= start && flat_index < end) {
      values[static_cast<std::size_t>(flat_index - start)] = value;
    }
  };

  if (tensor.ggml_type == 0) {
    for (std::uint64_t index = 0; index < count; index += 1) {
      values[static_cast<std::size_t>(index)] = read_f32_le(data + (start + index) * sizeof(float));
    }
    return;
  }
  if (tensor.ggml_type == 1) {
    for (std::uint64_t index = 0; index < count; index += 1) {
      values[static_cast<std::size_t>(index)] = half_to_float(read_u16_le(data + (start + index) * sizeof(std::uint16_t)));
    }
    return;
  }
  if (tensor.ggml_type == 30) {
    for (std::uint64_t index = 0; index < count; index += 1) {
      values[static_cast<std::size_t>(index)] = bfloat16_to_float(read_u16_le(data + (start + index) * sizeof(std::uint16_t)));
    }
    return;
  }

  const auto layout = ggml_type_layout(tensor.ggml_type);
  if (!layout.has_value()) {
    throw WorkerError(
      "NATIVE_EXECUTOR_UNIMPLEMENTED",
      "The native worker can index this tensor, but dequantization for its GGML type is not implemented yet.",
      JsonObject {
        {"tensor", JsonValue(tensor.name)},
        {"ggmlType", JsonValue(static_cast<std::uint64_t>(tensor.ggml_type))},
        {"ggmlTypeName", JsonValue(tensor.ggml_type_name.empty() ? "unknown" : tensor.ggml_type_name)},
      });
  }

  const std::uint64_t first_block = start / layout->block_size;
  const std::uint64_t last_block = (end - 1) / layout->block_size;

  if (tensor.ggml_type == 8) {
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const float delta = half_to_float(read_u16_le(block_ptr));
      const std::uint8_t *quants = block_ptr + sizeof(std::uint16_t);
      for (std::uint64_t item = 0; item < layout->block_size; item += 1) {
        const std::uint64_t flat_index = block * layout->block_size + item;
        if (flat_index >= tensor.element_count) {
          break;
        }
        if (flat_index < start || flat_index >= end) {
          continue;
        }
        const std::int8_t quant = static_cast<std::int8_t>(quants[item]);
        store_value(flat_index, delta * static_cast<float>(quant));
      }
    }
    return;
  }

  if (tensor.ggml_type == 12) {
    const auto get_scale_min_k4 = [](int j, const std::uint8_t *packed, std::uint8_t *scale, std::uint8_t *min) {
      if (j < 4) {
        *scale = packed[j] & 63;
        *min = packed[j + 4] & 63;
      } else {
        *scale = static_cast<std::uint8_t>((packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4));
        *min = static_cast<std::uint8_t>((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
      }
    };

    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const float d = half_to_float(read_u16_le(block_ptr));
      const float dmin = half_to_float(read_u16_le(block_ptr + sizeof(std::uint16_t)));
      const std::uint8_t *packed_scales = block_ptr + 2 * sizeof(std::uint16_t);
      const std::uint8_t *quants = packed_scales + k_q4_k_scale_bytes;

      int packed_index = 0;
      for (std::uint64_t chunk = 0; chunk < k_qk_k; chunk += 64) {
        std::uint8_t scale0 = 0;
        std::uint8_t min0 = 0;
        std::uint8_t scale1 = 0;
        std::uint8_t min1 = 0;
        get_scale_min_k4(packed_index + 0, packed_scales, &scale0, &min0);
        get_scale_min_k4(packed_index + 1, packed_scales, &scale1, &min1);
        const float d0 = d * static_cast<float>(scale0);
        const float m0 = dmin * static_cast<float>(min0);
        const float d1 = d * static_cast<float>(scale1);
        const float m1 = dmin * static_cast<float>(min1);
        const std::uint8_t *chunk_ptr = quants + (chunk / 2);

        for (std::uint64_t item = 0; item < 32; item += 1) {
          const std::uint64_t low_index = block * k_qk_k + chunk + item;
          const std::uint64_t high_index = low_index + 32;
          const std::uint8_t packed = chunk_ptr[item];
          if (low_index < tensor.element_count) {
            store_value(low_index, d0 * static_cast<float>(packed & 0x0F) - m0);
          }
          if (high_index < tensor.element_count) {
            store_value(high_index, d1 * static_cast<float>(packed >> 4) - m1);
          }
        }
        packed_index += 2;
      }
    }
    return;
  }

  if (tensor.ggml_type == 13) {
    const auto get_scale_min_k4 = [](int j, const std::uint8_t *packed, std::uint8_t *scale, std::uint8_t *min) {
      if (j < 4) {
        *scale = packed[j] & 63;
        *min = packed[j + 4] & 63;
      } else {
        *scale = static_cast<std::uint8_t>((packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4));
        *min = static_cast<std::uint8_t>((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
      }
    };

    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const float d = half_to_float(read_u16_le(block_ptr));
      const float dmin = half_to_float(read_u16_le(block_ptr + sizeof(std::uint16_t)));
      const std::uint8_t *packed_scales = block_ptr + 2 * sizeof(std::uint16_t);
      const std::uint8_t *qh = packed_scales + k_q4_k_scale_bytes;
      const std::uint8_t *ql = qh + (k_qk_k / 8);

      int packed_index = 0;
      std::uint8_t high_bit_low = 1;
      std::uint8_t high_bit_high = 2;
      for (std::uint64_t chunk = 0; chunk < k_qk_k; chunk += 64) {
        std::uint8_t scale0 = 0;
        std::uint8_t min0 = 0;
        std::uint8_t scale1 = 0;
        std::uint8_t min1 = 0;
        get_scale_min_k4(packed_index + 0, packed_scales, &scale0, &min0);
        get_scale_min_k4(packed_index + 1, packed_scales, &scale1, &min1);
        const float d0 = d * static_cast<float>(scale0);
        const float m0 = dmin * static_cast<float>(min0);
        const float d1 = d * static_cast<float>(scale1);
        const float m1 = dmin * static_cast<float>(min1);
        const std::uint8_t *chunk_ptr = ql + (chunk / 2);

        for (std::uint64_t item = 0; item < 32; item += 1) {
          const std::uint64_t low_index = block * k_qk_k + chunk + item;
          const std::uint64_t high_index = low_index + 32;
          const std::uint8_t packed = chunk_ptr[item];
          if (low_index < tensor.element_count) {
            store_value(low_index, d0 * static_cast<float>((packed & 0x0F) + ((qh[item] & high_bit_low) != 0 ? 16 : 0)) - m0);
          }
          if (high_index < tensor.element_count) {
            store_value(high_index, d1 * static_cast<float>((packed >> 4) + ((qh[item] & high_bit_high) != 0 ? 16 : 0)) - m1);
          }
        }
        packed_index += 2;
        high_bit_low <<= 2;
        high_bit_high <<= 2;
      }
    }
    return;
  }

  if (tensor.ggml_type == 14) {
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const std::uint8_t *ql = block_ptr;
      const std::uint8_t *qh = ql + (k_qk_k / 2);
      const std::int8_t *scales = reinterpret_cast<const std::int8_t *>(qh + (k_qk_k / 4));
      const float d = half_to_float(read_u16_le(block_ptr + layout->bytes_per_block - sizeof(std::uint16_t)));

      for (std::uint64_t chunk = 0; chunk < k_qk_k; chunk += 128) {
        const std::uint8_t *chunk_ql = ql + (chunk / 2);
        const std::uint8_t *chunk_qh = qh + (chunk / 4);
        const std::int8_t *chunk_scales = scales + (chunk / 16);
        for (std::uint64_t item = 0; item < 32; item += 1) {
          const int scale_index = static_cast<int>(item / 16);
          const std::int8_t q1 = static_cast<std::int8_t>(
            (chunk_ql[item] & 0x0F) | (((chunk_qh[item] >> 0) & 0x03) << 4)) - 32;
          const std::int8_t q2 = static_cast<std::int8_t>(
            (chunk_ql[item + 32] & 0x0F) | (((chunk_qh[item] >> 2) & 0x03) << 4)) - 32;
          const std::int8_t q3 = static_cast<std::int8_t>(
            (chunk_ql[item] >> 4) | (((chunk_qh[item] >> 4) & 0x03) << 4)) - 32;
          const std::int8_t q4 = static_cast<std::int8_t>(
            (chunk_ql[item + 32] >> 4) | (((chunk_qh[item] >> 6) & 0x03) << 4)) - 32;
          const std::uint64_t index0 = block * k_qk_k + chunk + item;
          const std::uint64_t index1 = index0 + 32;
          const std::uint64_t index2 = index0 + 64;
          const std::uint64_t index3 = index0 + 96;
          if (index0 < tensor.element_count) {
            store_value(index0, d * static_cast<float>(chunk_scales[scale_index + 0]) * static_cast<float>(q1));
          }
          if (index1 < tensor.element_count) {
            store_value(index1, d * static_cast<float>(chunk_scales[scale_index + 2]) * static_cast<float>(q2));
          }
          if (index2 < tensor.element_count) {
            store_value(index2, d * static_cast<float>(chunk_scales[scale_index + 4]) * static_cast<float>(q3));
          }
          if (index3 < tensor.element_count) {
            store_value(index3, d * static_cast<float>(chunk_scales[scale_index + 6]) * static_cast<float>(q4));
          }
        }
      }
    }
    return;
  }

  if (tensor.ggml_type == 2) {
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const float d = half_to_float(read_u16_le(block_ptr));
      const std::uint8_t *quants = block_ptr + sizeof(std::uint16_t);
      for (std::uint64_t item = 0; item < 16; item += 1) {
        const std::uint64_t low_index = block * layout->block_size + item;
        const std::uint64_t high_index = low_index + 16;
        const std::uint8_t packed = quants[item];
        if (low_index < tensor.element_count) {
          store_value(low_index, d * (static_cast<float>(packed & 0x0F) - 8.0f));
        }
        if (high_index < tensor.element_count) {
          store_value(high_index, d * (static_cast<float>(packed >> 4) - 8.0f));
        }
      }
    }
    return;
  }

  if (tensor.ggml_type == 10) {
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const std::uint8_t *scales = block_ptr;
      const std::uint8_t *quants = block_ptr + (k_qk_k / 16);
      const float d = half_to_float(read_u16_le(block_ptr + 80));
      const float dmin = half_to_float(read_u16_le(block_ptr + 82));
      std::uint64_t position = block * k_qk_k;
      int scale_index = 0;
      for (std::uint64_t half = 0; half < k_qk_k; half += 128) {
        const std::uint8_t *q = quants + (half / 4);
        for (int shift = 0; shift < 8; shift += 2) {
          for (int group = 0; group < 2; group += 1) {
            const std::uint8_t packed_scale = scales[scale_index];
            scale_index += 1;
            const float dl = d * static_cast<float>(packed_scale & 0x0F);
            const float ml = dmin * static_cast<float>(packed_scale >> 4);
            for (int item = 0; item < 16; item += 1) {
              if (position < tensor.element_count) {
                store_value(position, dl * static_cast<float>((q[item + group * 16] >> shift) & 3) - ml);
              }
              position += 1;
            }
          }
        }
      }
    }
    return;
  }

  if (tensor.ggml_type == 11) {
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const std::uint8_t *hmask = block_ptr;
      const std::uint8_t *quants = block_ptr + (k_qk_k / 8);
      const std::uint8_t *packed_scales = block_ptr + (k_qk_k / 8) + (k_qk_k / 4);
      const float d = half_to_float(read_u16_le(block_ptr + layout->bytes_per_block - sizeof(std::uint16_t)));
      // Unpack the sixteen 6-bit sub-block scales from their 12-byte packing:
      // low 4 bits come from bytes 0-7 (both nibbles), high 2 bits from bytes 8-11.
      int sub_scales[16];
      for (int j = 0; j < 16; j += 1) {
        const int low4 = j < 8
          ? (packed_scales[j] & 0x0F)
          : (packed_scales[j - 8] >> 4);
        const int high2 = (packed_scales[8 + (j % 4)] >> (2 * (j / 4))) & 0x03;
        sub_scales[j] = low4 | (high2 << 4);
      }
      std::uint64_t position = block * k_qk_k;
      int scale_index = 0;
      std::uint8_t mask_bit = 1;
      for (std::uint64_t half = 0; half < k_qk_k; half += 128) {
        const std::uint8_t *q = quants + (half / 4);
        for (int shift = 0; shift < 8; shift += 2) {
          for (int group = 0; group < 2; group += 1) {
            const float dl = d * static_cast<float>(sub_scales[scale_index] - 32);
            scale_index += 1;
            for (int item = 0; item < 16; item += 1) {
              const int source = item + group * 16;
              int q_value = (q[source] >> shift) & 3;
              if ((hmask[source] & mask_bit) == 0) {
                q_value -= 4;
              }
              if (position < tensor.element_count) {
                store_value(position, dl * static_cast<float>(q_value));
              }
              position += 1;
            }
          }
          mask_bit = static_cast<std::uint8_t>(mask_bit << 1);
        }
      }
    }
    return;
  }

  if (tensor.ggml_type == 23) {
    static const std::int8_t k_iq4_nl_values[16] = {
      -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const float d = half_to_float(read_u16_le(block_ptr));
      const std::uint32_t scales_h = read_u16_le(block_ptr + 2);
      const std::uint8_t *scales_l = block_ptr + 4;
      const std::uint8_t *quants = block_ptr + 8;
      for (int sub = 0; sub < 8; sub += 1) {
        const int ls = ((scales_l[sub / 2] >> (4 * (sub % 2))) & 0x0F)
          | (static_cast<int>((scales_h >> (2 * sub)) & 0x03) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        const std::uint8_t *q = quants + sub * 16;
        for (int item = 0; item < 16; item += 1) {
          const std::uint64_t low_index = block * k_qk_k + static_cast<std::uint64_t>(sub) * 32 + item;
          const std::uint64_t high_index = low_index + 16;
          if (low_index < tensor.element_count) {
            store_value(low_index, dl * static_cast<float>(k_iq4_nl_values[q[item] & 0x0F]));
          }
          if (high_index < tensor.element_count) {
            store_value(high_index, dl * static_cast<float>(k_iq4_nl_values[q[item] >> 4]));
          }
        }
      }
    }
    return;
  }

  if (tensor.ggml_type == 34) {
    // Ternary 1.69 bpw: 48 base-243 packed bytes (5 trits each), then 4 qh
    // bytes (4 trits each), scale at the end of the 54-byte block.
    static const std::uint8_t k_pow3[6] = {1, 3, 9, 27, 81, 243};
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const std::uint8_t *qs = block_ptr;
      const std::uint8_t *qh = block_ptr + 48;
      const float d = half_to_float(read_u16_le(block_ptr + 52));
      std::uint64_t position = block * k_qk_k;
      const auto emit_trit = [&](std::uint8_t packed, int digit) {
        const std::uint8_t scaled = static_cast<std::uint8_t>(packed * k_pow3[digit]);
        const int trit = static_cast<int>((static_cast<std::uint16_t>(scaled) * 3) >> 8) - 1;
        if (position < tensor.element_count) {
          store_value(position, static_cast<float>(trit) * d);
        }
        position += 1;
      };
      for (int digit = 0; digit < 5; digit += 1) {
        for (int item = 0; item < 32; item += 1) {
          emit_trit(qs[item], digit);
        }
      }
      for (int digit = 0; digit < 5; digit += 1) {
        for (int item = 0; item < 16; item += 1) {
          emit_trit(qs[32 + item], digit);
        }
      }
      for (int digit = 0; digit < 4; digit += 1) {
        for (int item = 0; item < 4; item += 1) {
          emit_trit(qh[item], digit);
        }
      }
    }
    return;
  }

  if (tensor.ggml_type == 35) {
    // Ternary 2.06 bpw: 64 bytes of 2-bit quants (values 0..2 meaning -1..1).
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const std::uint8_t *qs = block_ptr;
      const float d = half_to_float(read_u16_le(block_ptr + 64));
      std::uint64_t position = block * k_qk_k;
      for (int chunk = 0; chunk < 64; chunk += 32) {
        for (int shift = 0; shift < 8; shift += 2) {
          for (int item = 0; item < 32; item += 1) {
            const int q_value = static_cast<int>((qs[chunk + item] >> shift) & 3) - 1;
            if (position < tensor.element_count) {
              store_value(position, static_cast<float>(q_value) * d);
            }
            position += 1;
          }
        }
      }
    }
    return;
  }

  throw WorkerError(
    "NATIVE_EXECUTOR_UNIMPLEMENTED",
    "The native worker can index this tensor, but dequantization for its GGML type is not implemented yet.",
    JsonObject {
      {"tensor", JsonValue(tensor.name)},
      {"ggmlType", JsonValue(static_cast<std::uint64_t>(tensor.ggml_type))},
      {"ggmlTypeName", JsonValue(tensor.ggml_type_name.empty() ? "unknown" : tensor.ggml_type_name)},
    });
}

static std::vector<float> dequantize_tensor_f32(const LoadedModelState &state, const TensorInfo &tensor) {
  std::vector<float> values;
  values.reserve(static_cast<std::size_t>(tensor.element_count));
  const std::uint8_t *data = tensor_data_ptr(state, tensor);

  if (tensor.ggml_type == 0) {
    for (std::uint64_t index = 0; index < tensor.element_count; index += 1) {
      values.push_back(read_f32_le(data + index * sizeof(float)));
    }
    return values;
  }
  if (tensor.ggml_type == 1) {
    for (std::uint64_t index = 0; index < tensor.element_count; index += 1) {
      values.push_back(half_to_float(read_u16_le(data + index * sizeof(std::uint16_t))));
    }
    return values;
  }
  if (tensor.ggml_type == 30) {
    for (std::uint64_t index = 0; index < tensor.element_count; index += 1) {
      values.push_back(bfloat16_to_float(read_u16_le(data + index * sizeof(std::uint16_t))));
    }
    return values;
  }
  if (tensor.ggml_type == 8) {
    const auto layout = ggml_type_layout(tensor.ggml_type);
    if (!layout.has_value()) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Missing GGML type layout for Q8_0 tensor.");
    }
    const std::uint64_t block_size = layout->block_size;
    const std::uint64_t block_count = (tensor.element_count + block_size - 1) / block_size;
    values.resize(static_cast<std::size_t>(tensor.element_count));
    for (std::uint64_t block = 0; block < block_count; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const float delta = half_to_float(read_u16_le(block_ptr));
      const std::uint8_t *quants = block_ptr + sizeof(std::uint16_t);
      for (std::uint64_t item = 0; item < block_size; item += 1) {
        const std::uint64_t value_index = block * block_size + item;
        if (value_index >= tensor.element_count) {
          break;
        }
        const std::int8_t quant = static_cast<std::int8_t>(quants[item]);
        values[static_cast<std::size_t>(value_index)] = delta * static_cast<float>(quant);
      }
    }
    return values;
  }
  if (tensor.ggml_type == 12) {
    const auto layout = ggml_type_layout(tensor.ggml_type);
    if (!layout.has_value()) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Missing GGML type layout for Q4_K tensor.");
    }
    const auto get_scale_min_k4 = [](int j, const std::uint8_t *packed, std::uint8_t *scale, std::uint8_t *min) {
      if (j < 4) {
        *scale = packed[j] & 63;
        *min = packed[j + 4] & 63;
      } else {
        *scale = static_cast<std::uint8_t>((packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4));
        *min = static_cast<std::uint8_t>((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
      }
    };

    const std::uint64_t block_count = (tensor.element_count + layout->block_size - 1) / layout->block_size;
    values.resize(static_cast<std::size_t>(tensor.element_count), 0.0f);
    for (std::uint64_t block = 0; block < block_count; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const float d = half_to_float(read_u16_le(block_ptr));
      const float dmin = half_to_float(read_u16_le(block_ptr + sizeof(std::uint16_t)));
      const std::uint8_t *packed_scales = block_ptr + 2 * sizeof(std::uint16_t);
      const std::uint8_t *quants = packed_scales + k_q4_k_scale_bytes;

      int packed_index = 0;
      for (std::uint64_t chunk = 0; chunk < k_qk_k; chunk += 64) {
        std::uint8_t scale0 = 0;
        std::uint8_t min0 = 0;
        std::uint8_t scale1 = 0;
        std::uint8_t min1 = 0;
        get_scale_min_k4(packed_index + 0, packed_scales, &scale0, &min0);
        get_scale_min_k4(packed_index + 1, packed_scales, &scale1, &min1);
        const float d0 = d * static_cast<float>(scale0);
        const float m0 = dmin * static_cast<float>(min0);
        const float d1 = d * static_cast<float>(scale1);
        const float m1 = dmin * static_cast<float>(min1);
        const std::uint8_t *chunk_ptr = quants + (chunk / 2);

        for (std::uint64_t item = 0; item < 32; item += 1) {
          const std::uint64_t low_index = block * k_qk_k + chunk + item;
          const std::uint64_t high_index = low_index + 32;
          const std::uint8_t packed = chunk_ptr[item];
          if (low_index < tensor.element_count) {
            values[static_cast<std::size_t>(low_index)] = d0 * static_cast<float>(packed & 0x0F) - m0;
          }
          if (high_index < tensor.element_count) {
            values[static_cast<std::size_t>(high_index)] = d1 * static_cast<float>(packed >> 4) - m1;
          }
        }
        packed_index += 2;
      }
    }
    return values;
  }
  if (tensor.ggml_type == 13) {
    const auto layout = ggml_type_layout(tensor.ggml_type);
    if (!layout.has_value()) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Missing GGML type layout for Q5_K tensor.");
    }
    const auto get_scale_min_k4 = [](int j, const std::uint8_t *packed, std::uint8_t *scale, std::uint8_t *min) {
      if (j < 4) {
        *scale = packed[j] & 63;
        *min = packed[j + 4] & 63;
      } else {
        *scale = static_cast<std::uint8_t>((packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4));
        *min = static_cast<std::uint8_t>((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
      }
    };

    const std::uint64_t block_count = (tensor.element_count + layout->block_size - 1) / layout->block_size;
    values.resize(static_cast<std::size_t>(tensor.element_count), 0.0f);
    for (std::uint64_t block = 0; block < block_count; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const float d = half_to_float(read_u16_le(block_ptr));
      const float dmin = half_to_float(read_u16_le(block_ptr + sizeof(std::uint16_t)));
      const std::uint8_t *packed_scales = block_ptr + 2 * sizeof(std::uint16_t);
      const std::uint8_t *qh = packed_scales + k_q4_k_scale_bytes;
      const std::uint8_t *ql = qh + (k_qk_k / 8);

      int packed_index = 0;
      std::uint8_t high_bit_low = 1;
      std::uint8_t high_bit_high = 2;
      for (std::uint64_t chunk = 0; chunk < k_qk_k; chunk += 64) {
        std::uint8_t scale0 = 0;
        std::uint8_t min0 = 0;
        std::uint8_t scale1 = 0;
        std::uint8_t min1 = 0;
        get_scale_min_k4(packed_index + 0, packed_scales, &scale0, &min0);
        get_scale_min_k4(packed_index + 1, packed_scales, &scale1, &min1);
        const float d0 = d * static_cast<float>(scale0);
        const float m0 = dmin * static_cast<float>(min0);
        const float d1 = d * static_cast<float>(scale1);
        const float m1 = dmin * static_cast<float>(min1);
        const std::uint8_t *chunk_ptr = ql + (chunk / 2);

        for (std::uint64_t item = 0; item < 32; item += 1) {
          const std::uint64_t low_index = block * k_qk_k + chunk + item;
          const std::uint64_t high_index = low_index + 32;
          const std::uint8_t packed = chunk_ptr[item];
          if (low_index < tensor.element_count) {
            values[static_cast<std::size_t>(low_index)] = d0 * static_cast<float>((packed & 0x0F) + ((qh[item] & high_bit_low) != 0 ? 16 : 0)) - m0;
          }
          if (high_index < tensor.element_count) {
            values[static_cast<std::size_t>(high_index)] = d1 * static_cast<float>((packed >> 4) + ((qh[item] & high_bit_high) != 0 ? 16 : 0)) - m1;
          }
        }
        packed_index += 2;
        high_bit_low <<= 2;
        high_bit_high <<= 2;
      }
    }
    return values;
  }
  if (tensor.ggml_type == 14) {
    const auto layout = ggml_type_layout(tensor.ggml_type);
    if (!layout.has_value()) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Missing GGML type layout for Q6_K tensor.");
    }
    const std::uint64_t block_count = (tensor.element_count + layout->block_size - 1) / layout->block_size;
    values.resize(static_cast<std::size_t>(tensor.element_count), 0.0f);
    for (std::uint64_t block = 0; block < block_count; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const std::uint8_t *ql = block_ptr;
      const std::uint8_t *qh = ql + (k_qk_k / 2);
      const std::int8_t *scales = reinterpret_cast<const std::int8_t *>(qh + (k_qk_k / 4));
      const float d = half_to_float(read_u16_le(block_ptr + layout->bytes_per_block - sizeof(std::uint16_t)));

      for (std::uint64_t chunk = 0; chunk < k_qk_k; chunk += 128) {
        const std::uint8_t *chunk_ql = ql + (chunk / 2);
        const std::uint8_t *chunk_qh = qh + (chunk / 4);
        const std::int8_t *chunk_scales = scales + (chunk / 16);
        for (std::uint64_t item = 0; item < 32; item += 1) {
          const int scale_index = static_cast<int>(item / 16);
          const std::int8_t q1 = static_cast<std::int8_t>(
            (chunk_ql[item] & 0x0F) | (((chunk_qh[item] >> 0) & 0x03) << 4)) - 32;
          const std::int8_t q2 = static_cast<std::int8_t>(
            (chunk_ql[item + 32] & 0x0F) | (((chunk_qh[item] >> 2) & 0x03) << 4)) - 32;
          const std::int8_t q3 = static_cast<std::int8_t>(
            (chunk_ql[item] >> 4) | (((chunk_qh[item] >> 4) & 0x03) << 4)) - 32;
          const std::int8_t q4 = static_cast<std::int8_t>(
            (chunk_ql[item + 32] >> 4) | (((chunk_qh[item] >> 6) & 0x03) << 4)) - 32;
          const std::uint64_t index0 = block * k_qk_k + chunk + item;
          const std::uint64_t index1 = index0 + 32;
          const std::uint64_t index2 = index0 + 64;
          const std::uint64_t index3 = index0 + 96;
          if (index0 < tensor.element_count) {
            values[static_cast<std::size_t>(index0)] = d * static_cast<float>(chunk_scales[scale_index + 0]) * static_cast<float>(q1);
          }
          if (index1 < tensor.element_count) {
            values[static_cast<std::size_t>(index1)] = d * static_cast<float>(chunk_scales[scale_index + 2]) * static_cast<float>(q2);
          }
          if (index2 < tensor.element_count) {
            values[static_cast<std::size_t>(index2)] = d * static_cast<float>(chunk_scales[scale_index + 4]) * static_cast<float>(q3);
          }
          if (index3 < tensor.element_count) {
            values[static_cast<std::size_t>(index3)] = d * static_cast<float>(chunk_scales[scale_index + 6]) * static_cast<float>(q4);
          }
        }
      }
    }
    return values;
  }

  // Types without a dedicated full-tensor kernel (Q4_0, Q2_K, Q3_K, IQ4_XS,
  // TQ1_0, TQ2_0) share the range-based implementation.
  std::vector<float> range_values;
  dequantize_tensor_range_into_f32(state, tensor, 0, tensor.element_count, range_values);
  return range_values;
}

static std::vector<float> tensor_row_f32(const LoadedModelState &state, const TensorInfo &tensor, std::uint64_t row_index) {
  if (tensor.dimensions.size() < 2) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Tensor row access requires at least two dimensions.");
  }
  const std::uint64_t row_width = tensor.dimensions[0];
  const std::uint64_t row_count = tensor.element_count / row_width;
  if (row_width == 0 || row_index >= row_count) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Requested tensor row is out of bounds.");
  }
  std::vector<float> row;
  dequantize_tensor_range_into_f32(state, tensor, row_index * row_width, row_width, row);
  return row;
}

static std::vector<float> tensor_vector_f32(const LoadedModelState &state, const TensorInfo &tensor) {
  if (tensor.dimensions.size() != 1) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Vector tensor access requires a one-dimensional tensor.");
  }
  return dequantize_tensor_f32(state, tensor);
}

static void cache_vector_tensor_f32(LoadedModelState &state, std::size_t index) {
  if (index == k_missing_tensor_index || state.vector_tensor_cache.find(index) != state.vector_tensor_cache.end()) {
    return;
  }
  const TensorInfo &tensor = tensor_at(state, index);
  if (tensor.dimensions.size() != 1) {
    return;
  }
  state.vector_tensor_cache.emplace(index, tensor_vector_f32(state, tensor));
}

static void build_vector_tensor_cache(LoadedModelState &state) {
  cache_vector_tensor_f32(state, state.tensor_index.output_norm);
  cache_vector_tensor_f32(state, state.tensor_index.output_norm_bias);
  cache_vector_tensor_f32(state, state.tensor_index.output_bias);
  cache_vector_tensor_f32(state, state.tensor_index.per_layer_projection_norm);
  cache_vector_tensor_f32(state, state.tensor_index.rope_factors_long);
  cache_vector_tensor_f32(state, state.tensor_index.rope_factors_short);
  if (state.tensor_index.rope_freqs != k_missing_tensor_index) {
    cache_vector_tensor_f32(state, state.tensor_index.rope_freqs);
  }
  for (const LlamaLayerTensorIndex &layer : state.tensor_index.layers) {
    cache_vector_tensor_f32(state, layer.attention_norm);
    cache_vector_tensor_f32(state, layer.attention_norm_bias);
    cache_vector_tensor_f32(state, layer.query_norm);
    cache_vector_tensor_f32(state, layer.key_norm);
    cache_vector_tensor_f32(state, layer.ffn_norm);
    cache_vector_tensor_f32(state, layer.ffn_norm_bias);
    cache_vector_tensor_f32(state, layer.post_attention_norm);
    cache_vector_tensor_f32(state, layer.post_ffw_norm);
    cache_vector_tensor_f32(state, layer.post_norm);
    cache_vector_tensor_f32(state, layer.layer_output_scale);
    cache_vector_tensor_f32(state, layer.qkv_fused_bias);
    cache_vector_tensor_f32(state, layer.query_bias);
    cache_vector_tensor_f32(state, layer.key_bias);
    cache_vector_tensor_f32(state, layer.value_bias);
    cache_vector_tensor_f32(state, layer.output_bias);
    cache_vector_tensor_f32(state, layer.up_bias);
    cache_vector_tensor_f32(state, layer.down_bias);
    cache_vector_tensor_f32(state, layer.shexp_gate_inp);
  }
}

static const std::vector<float> &cached_tensor_vector_f32(
  const LoadedModelState &state,
  std::size_t index,
  const std::string &label) {
  if (index == k_missing_tensor_index || index >= state.model.tensors.size()) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " vector tensor is unavailable.");
  }
  const auto found = state.vector_tensor_cache.find(index);
  if (found != state.vector_tensor_cache.end()) {
    return found->second;
  }
  const TensorInfo &tensor = tensor_at(state, index);
  if (tensor.dimensions.size() != 1) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " requires a one-dimensional tensor.");
  }
  const auto inserted = state.vector_tensor_cache.emplace(index, tensor_vector_f32(state, tensor));
  return inserted.first->second;
}

static std::vector<float> rms_norm(
  const std::vector<float> &input,
  const std::vector<float> &weights,
  double epsilon = 1e-5) {
  if (input.size() != weights.size()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "RMSNorm requires input and weight vectors with the same size.");
  }
  double sum = 0.0;
  for (float value : input) {
    sum += static_cast<double>(value) * static_cast<double>(value);
  }
  const double mean = input.empty() ? 0.0 : sum / static_cast<double>(input.size());
  const float scale = static_cast<float>(1.0 / std::sqrt(mean + epsilon));
  std::vector<float> output(input.size(), 0.0f);
  for (std::size_t index = 0; index < input.size(); index += 1) {
    output[index] = input[index] * scale * weights[index];
  }
  return output;
}

static std::vector<float> rms_norm_heads(
  const std::vector<float> &input,
  const std::vector<float> &weights,
  std::uint64_t head_count,
  std::uint64_t head_dim,
  const std::string &label,
  double epsilon = 1e-5) {
  if (input.size() != static_cast<std::size_t>(head_count * head_dim) || weights.size() != static_cast<std::size_t>(head_dim)) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " requires per-head input vectors and one norm vector per head dimension.");
  }
  std::vector<float> output(input.size(), 0.0f);
  for (std::uint64_t head = 0; head < head_count; head += 1) {
    const std::size_t offset = static_cast<std::size_t>(head * head_dim);
    double sum = 0.0;
    for (std::uint64_t item = 0; item < head_dim; item += 1) {
      const float value = input[offset + static_cast<std::size_t>(item)];
      sum += static_cast<double>(value) * static_cast<double>(value);
    }
    const double mean = head_dim == 0 ? 0.0 : sum / static_cast<double>(head_dim);
    const float scale = static_cast<float>(1.0 / std::sqrt(mean + epsilon));
    for (std::uint64_t item = 0; item < head_dim; item += 1) {
      const std::size_t index = offset + static_cast<std::size_t>(item);
      output[index] = input[index] * scale * weights[static_cast<std::size_t>(item)];
    }
  }
  return output;
}

static std::vector<float> rms_norm_heads_unweighted(
  const std::vector<float> &input,
  std::uint64_t head_count,
  std::uint64_t head_dim,
  const std::string &label,
  double epsilon = 1e-5) {
  if (input.size() != static_cast<std::size_t>(head_count * head_dim)) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " requires per-head input vectors.");
  }
  std::vector<float> output(input.size(), 0.0f);
  for (std::uint64_t head = 0; head < head_count; head += 1) {
    const std::size_t offset = static_cast<std::size_t>(head * head_dim);
    double sum = 0.0;
    for (std::uint64_t item = 0; item < head_dim; item += 1) {
      const float value = input[offset + static_cast<std::size_t>(item)];
      sum += static_cast<double>(value) * static_cast<double>(value);
    }
    const double mean = head_dim == 0 ? 0.0 : sum / static_cast<double>(head_dim);
    const float scale = static_cast<float>(1.0 / std::sqrt(mean + epsilon));
    for (std::uint64_t item = 0; item < head_dim; item += 1) {
      const std::size_t index = offset + static_cast<std::size_t>(item);
      output[index] = input[index] * scale;
    }
  }
  return output;
}

// Classic LayerNorm (mean/variance with weight and optional bias) for the
// phi and starcoder2 family graphs; the RMS-norm families keep rms_norm.
static std::vector<float> layer_norm(
  const std::vector<float> &input,
  const std::vector<float> &weights,
  const std::vector<float> *bias,
  double epsilon = 1e-5) {
  if (input.size() != weights.size() || (bias != nullptr && bias->size() != input.size())) {
    throw WorkerError("NATIVE_WORKER_ERROR", "LayerNorm requires input, weight, and bias vectors with the same size.");
  }
  double sum = 0.0;
  for (float value : input) {
    sum += static_cast<double>(value);
  }
  const double mean = input.empty() ? 0.0 : sum / static_cast<double>(input.size());
  double variance_sum = 0.0;
  for (float value : input) {
    const double centered = static_cast<double>(value) - mean;
    variance_sum += centered * centered;
  }
  const double variance = input.empty() ? 0.0 : variance_sum / static_cast<double>(input.size());
  const float scale = static_cast<float>(1.0 / std::sqrt(variance + epsilon));
  std::vector<float> output(input.size(), 0.0f);
  for (std::size_t index = 0; index < input.size(); index += 1) {
    const float normalized = (input[index] - static_cast<float>(mean)) * scale;
    output[index] = normalized * weights[index] + (bias != nullptr ? (*bias)[index] : 0.0f);
  }
  return output;
}

static void add_bias_in_place(std::vector<float> &values, const std::vector<float> &bias, const std::string &label) {
  if (values.size() != bias.size()) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " bias size does not match the projection output.");
  }
  for (std::size_t index = 0; index < values.size(); index += 1) {
    values[index] += bias[index];
  }
}

static MatrixShape tensor_matrix_shape(const TensorInfo &tensor, const std::string &label) {
  if (tensor.dimensions.size() < 2) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " requires a tensor with at least two dimensions.");
  }
  const std::uint64_t row_width = tensor.dimensions[0];
  const std::uint64_t row_count = row_width == 0 ? 0 : (tensor.element_count / row_width);
  if (row_width == 0 || row_count == 0) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " has an invalid tensor shape.");
  }
  return MatrixShape {row_width, row_count};
}

static std::vector<float> matrix_vector_mul(
  const std::vector<float> &matrix,
  const MatrixShape &shape,
  const std::vector<float> &input,
  const std::string &label) {
  if (shape.row_width != input.size()) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " width does not match the input vector size.");
  }
  if (matrix.size() != static_cast<std::size_t>(shape.row_width * shape.row_count)) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " matrix payload does not match the advertised tensor shape.");
  }
  std::vector<float> output(static_cast<std::size_t>(shape.row_count), 0.0f);
  for (std::uint64_t row = 0; row < shape.row_count; row += 1) {
    double dot = 0.0;
    const std::size_t offset = static_cast<std::size_t>(row * shape.row_width);
    for (std::size_t index = 0; index < input.size(); index += 1) {
      dot += static_cast<double>(matrix[offset + index]) * static_cast<double>(input[index]);
    }
    output[static_cast<std::size_t>(row)] = static_cast<float>(dot);
  }
  return output;
}

static double tensor_range_dot_f32(
  const LoadedModelState &state,
  const TensorInfo &tensor,
  std::uint64_t start,
  std::uint64_t count,
  const std::vector<float> &input,
  const std::string &label) {
  if (count != input.size()) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " tensor range width does not match the input vector size.");
  }
  if (start > tensor.element_count || count > tensor.element_count - start) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " tensor range is out of bounds.");
  }
  if (count == 0) {
    return 0.0;
  }

  const std::uint64_t end = start + count;
  const std::uint8_t *data = tensor_data_ptr(state, tensor);
  double dot = 0.0;
  const auto accumulate = [&](std::uint64_t flat_index, float value) {
    if (flat_index >= start && flat_index < end) {
      dot += static_cast<double>(value) * static_cast<double>(input[static_cast<std::size_t>(flat_index - start)]);
    }
  };

  if (tensor.ggml_type == 0) {
    for (std::uint64_t index = 0; index < count; index += 1) {
      dot += static_cast<double>(read_f32_le(data + (start + index) * sizeof(float)))
        * static_cast<double>(input[static_cast<std::size_t>(index)]);
    }
    return dot;
  }
  if (tensor.ggml_type == 1) {
    for (std::uint64_t index = 0; index < count; index += 1) {
      dot += static_cast<double>(half_to_float(read_u16_le(data + (start + index) * sizeof(std::uint16_t))))
        * static_cast<double>(input[static_cast<std::size_t>(index)]);
    }
    return dot;
  }
  if (tensor.ggml_type == 30) {
    for (std::uint64_t index = 0; index < count; index += 1) {
      dot += static_cast<double>(bfloat16_to_float(read_u16_le(data + (start + index) * sizeof(std::uint16_t))))
        * static_cast<double>(input[static_cast<std::size_t>(index)]);
    }
    return dot;
  }

  const auto layout = ggml_type_layout(tensor.ggml_type);
  if (!layout.has_value()) {
    std::vector<float> values;
    dequantize_tensor_range_into_f32(state, tensor, start, count, values);
    for (std::size_t index = 0; index < values.size(); index += 1) {
      dot += static_cast<double>(values[index]) * static_cast<double>(input[index]);
    }
    return dot;
  }

  const std::uint64_t first_block = start / layout->block_size;
  const std::uint64_t last_block = (end - 1) / layout->block_size;
  const bool aligned_full_blocks = (start % layout->block_size) == 0 && (count % layout->block_size) == 0;

  if (tensor.ggml_type == 2) {
    if (aligned_full_blocks) {
      for (std::uint64_t block = first_block; block <= last_block; block += 1) {
        const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
        const float d = half_to_float(read_u16_le(block_ptr));
        const std::uint8_t *quants = block_ptr + sizeof(std::uint16_t);
        const std::size_t input_offset = static_cast<std::size_t>((block - first_block) * layout->block_size);
        // Split accumulators, same reason as the Q4_K path below: a single f64
        // running total makes every multiply-add depend on the previous one,
        // which cannot vectorise. Measured +49% on Q4_K.
        float low_sum = 0.0f;
        float high_sum = 0.0f;
        for (std::uint64_t item = 0; item < 16; item += 1) {
          const std::uint8_t packed = quants[item];
          const std::size_t low_index = input_offset + static_cast<std::size_t>(item);
          const std::size_t high_index = low_index + 16;
          low_sum += (d * (static_cast<float>(packed & 0x0F) - 8.0f)) * input[low_index];
          high_sum += (d * (static_cast<float>(packed >> 4) - 8.0f)) * input[high_index];
        }
        dot += static_cast<double>(low_sum) + static_cast<double>(high_sum);
      }
      return dot;
    }
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const float d = half_to_float(read_u16_le(block_ptr));
      const std::uint8_t *quants = block_ptr + sizeof(std::uint16_t);
      for (std::uint64_t item = 0; item < 16; item += 1) {
        const std::uint64_t low_index = block * layout->block_size + item;
        const std::uint64_t high_index = low_index + 16;
        const std::uint8_t packed = quants[item];
        if (low_index < tensor.element_count) {
          accumulate(low_index, d * (static_cast<float>(packed & 0x0F) - 8.0f));
        }
        if (high_index < tensor.element_count) {
          accumulate(high_index, d * (static_cast<float>(packed >> 4) - 8.0f));
        }
      }
    }
    return dot;
  }

  if (tensor.ggml_type == 8) {
    if (aligned_full_blocks) {
      for (std::uint64_t block = first_block; block <= last_block; block += 1) {
        const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
        const float delta = half_to_float(read_u16_le(block_ptr));
        const std::uint8_t *quants = block_ptr + sizeof(std::uint16_t);
        const std::size_t input_offset = static_cast<std::size_t>((block - first_block) * layout->block_size);
        // Two accumulators over the even/odd elements so the block's adds form
        // two independent chains instead of one; see the Q4_K path below.
        float even_sum = 0.0f;
        float odd_sum = 0.0f;
        const std::uint64_t pairs = layout->block_size & ~1ull;
        for (std::uint64_t item = 0; item < pairs; item += 2) {
          even_sum += (delta * static_cast<float>(static_cast<std::int8_t>(quants[item])))
            * input[input_offset + static_cast<std::size_t>(item)];
          odd_sum += (delta * static_cast<float>(static_cast<std::int8_t>(quants[item + 1])))
            * input[input_offset + static_cast<std::size_t>(item + 1)];
        }
        for (std::uint64_t item = pairs; item < layout->block_size; item += 1) {
          even_sum += (delta * static_cast<float>(static_cast<std::int8_t>(quants[item])))
            * input[input_offset + static_cast<std::size_t>(item)];
        }
        dot += static_cast<double>(even_sum) + static_cast<double>(odd_sum);
      }
      return dot;
    }
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const float delta = half_to_float(read_u16_le(block_ptr));
      const std::uint8_t *quants = block_ptr + sizeof(std::uint16_t);
      for (std::uint64_t item = 0; item < layout->block_size; item += 1) {
        const std::uint64_t flat_index = block * layout->block_size + item;
        if (flat_index >= tensor.element_count) {
          break;
        }
        if (flat_index < start || flat_index >= end) {
          continue;
        }
        accumulate(flat_index, delta * static_cast<float>(static_cast<std::int8_t>(quants[item])));
      }
    }
    return dot;
  }

  if (tensor.ggml_type == 12) {
    const auto get_scale_min_k4 = [](int j, const std::uint8_t *packed, std::uint8_t *scale, std::uint8_t *min) {
      if (j < 4) {
        *scale = packed[j] & 63;
        *min = packed[j + 4] & 63;
      } else {
        *scale = static_cast<std::uint8_t>((packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4));
        *min = static_cast<std::uint8_t>((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
      }
    };

    if (aligned_full_blocks) {
      for (std::uint64_t block = first_block; block <= last_block; block += 1) {
        const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
        const float d = half_to_float(read_u16_le(block_ptr));
        const float dmin = half_to_float(read_u16_le(block_ptr + sizeof(std::uint16_t)));
        const std::uint8_t *packed_scales = block_ptr + 2 * sizeof(std::uint16_t);
        const std::uint8_t *quants = packed_scales + k_q4_k_scale_bytes;
        const std::size_t input_block_offset = static_cast<std::size_t>((block - first_block) * k_qk_k);

        int packed_index = 0;
#ifdef LIZARD_NATIVE_AVX2
        // Four accumulators for the whole 256-weight block, folded once at the
        // end. Two things are going on here.
        //
        // Block scope rather than chunk scope: per-chunk folding cost four
        // horizontal reductions per block at roughly ten cycles each. The
        // scales differ per chunk but are applied to the weights before
        // accumulation, so a running vector stays correct across chunks.
        //
        // Four of them rather than one: with a single accumulator every FMA in
        // the item loop waits on the previous one to retire. FMA latency is
        // ~4 cycles and throughput is 2/cycle, so a serial chain leaves the
        // vector units idle ~87% of the time - 8 FMAs x 4 cycles = 32 cycles
        // per chunk, which is the ~2 GB/s this kernel was stuck at. Four
        // independent chains let the out-of-order engine keep both FMA ports
        // fed, and the sum is identical up to f32 re-association, which this
        // path already accepts (see the factoring note below).
        __m256 block_acc[4] = {
          _mm256_setzero_ps(), _mm256_setzero_ps(),
          _mm256_setzero_ps(), _mm256_setzero_ps()};
#endif
        for (std::uint64_t chunk = 0; chunk < k_qk_k; chunk += 64) {
          std::uint8_t scale0 = 0;
          std::uint8_t min0 = 0;
          std::uint8_t scale1 = 0;
          std::uint8_t min1 = 0;
          get_scale_min_k4(packed_index + 0, packed_scales, &scale0, &min0);
          get_scale_min_k4(packed_index + 1, packed_scales, &scale1, &min1);
          const float d0 = d * static_cast<float>(scale0);
          const float m0 = dmin * static_cast<float>(min0);
          const float d1 = d * static_cast<float>(scale1);
          const float m1 = dmin * static_cast<float>(min1);
          const std::uint8_t *chunk_ptr = quants + (chunk / 2);
          const std::size_t input_chunk_offset = input_block_offset + static_cast<std::size_t>(chunk);

          // Two independent f32 accumulators, folded into the f64 total once
          // per 64-weight chunk. The previous form added both halves straight
          // into `dot`, so every one of the 64 multiply-adds in this chunk
          // depended on the result of the one before it - a serial f64
          // dependency chain that no compiler can vectorise and that runs at
          // roughly one element per FMA latency. Splitting low/high removes the
          // dependency between the halves and lets the 32-iteration loop become
          // SIMD on the AVX2/AVX-512 builds this runner already ships.
          //
          // Precision is deliberately kept close to the original: the f32 sums
          // span 32 terms each before widening, and the running total stays f64.
          float low_sum = 0.0f;
          float high_sum = 0.0f;
#ifdef LIZARD_NATIVE_AVX2
          // sum((d*q - m) * x) factorises to d*sum(q*x) - m*sum(x), turning the
          // per-element dequantise-then-multiply into two independent
          // reductions the vector unit runs eight lanes at a time. This is the
          // shape llama.cpp's kernels use, and it is why they pull ~16 GB/s
          // from this bus where the scalar loop below manages ~1.6.
          //
          // Numerics: the factored form re-associates the sum, so results are
          // close but not bit-identical to the scalar path. The GPU lane is the
          // reference for decode; this is the fallback that every machine
          // without a working D3D12 lane runs on.
          {
            // One accumulator, folded once. Scaling inside the loop (weight =
            // q*d - m as a vector, then a single FMA against the input) keeps
            // the whole chunk in registers, where the earlier version folded
            // four vectors horizontally per 32 elements - about 192 horizontal
            // sums per row at roughly ten cycles each, which cost more than the
            // arithmetic they were reducing.
            const __m256i low_mask = _mm256_set1_epi32(0x0F);
            const __m256 d0_v = _mm256_set1_ps(d0);
            const __m256 m0_v = _mm256_set1_ps(m0);
            const __m256 d1_v = _mm256_set1_ps(d1);
            const __m256 m1_v = _mm256_set1_ps(m1);
            // Constant trip count and constant stride, so the compiler fully
            // unrolls this and the accumulator indices below resolve to fixed
            // registers - the low/high halves of successive iterations land on
            // four separate dependency chains.
            for (std::uint64_t item = 0; item < 32; item += 8) {
              const std::size_t lane = static_cast<std::size_t>((item >> 3) & 1);
              const __m128i packed_bytes = _mm_loadl_epi64(
                reinterpret_cast<const __m128i *>(chunk_ptr + item));
              const __m256i widened = _mm256_cvtepu8_epi32(packed_bytes);
              const __m256 q_low = _mm256_cvtepi32_ps(_mm256_and_si256(widened, low_mask));
              const __m256 q_high = _mm256_cvtepi32_ps(_mm256_srli_epi32(widened, 4));
              const float *const x_low =
                input.data() + input_chunk_offset + static_cast<std::size_t>(item);
              const __m256 in_low = _mm256_loadu_ps(x_low);
              const __m256 in_high = _mm256_loadu_ps(x_low + 32);
              const __m256 w_low = _mm256_fmsub_ps(q_low, d0_v, m0_v);
              const __m256 w_high = _mm256_fmsub_ps(q_high, d1_v, m1_v);
              block_acc[lane] = _mm256_fmadd_ps(w_low, in_low, block_acc[lane]);
              block_acc[2 + lane] = _mm256_fmadd_ps(w_high, in_high, block_acc[2 + lane]);
            }
          }
#else
          for (std::uint64_t item = 0; item < 32; item += 1) {
            const std::uint8_t packed = chunk_ptr[item];
            const std::size_t low_index = input_chunk_offset + static_cast<std::size_t>(item);
            const std::size_t high_index = low_index + 32;
            low_sum += (d0 * static_cast<float>(packed & 0x0F) - m0) * input[low_index];
            high_sum += (d1 * static_cast<float>(packed >> 4) - m1) * input[high_index];
          }
          dot += static_cast<double>(low_sum) + static_cast<double>(high_sum);
#endif
          (void)low_sum;
          (void)high_sum;
          packed_index += 2;
        }
#ifdef LIZARD_NATIVE_AVX2
        // One horizontal reduction per 256-weight block instead of four: the
        // four chains merge with vector adds (1 cycle each) before the single
        // expensive horizontal fold.
        const __m256 acc_sum = _mm256_add_ps(
          _mm256_add_ps(block_acc[0], block_acc[1]),
          _mm256_add_ps(block_acc[2], block_acc[3]));
        __m128 folded = _mm_add_ps(
          _mm256_castps256_ps128(acc_sum), _mm256_extractf128_ps(acc_sum, 1));
        folded = _mm_add_ps(folded, _mm_movehl_ps(folded, folded));
        folded = _mm_add_ss(folded, _mm_shuffle_ps(folded, folded, 0x55));
        dot += static_cast<double>(_mm_cvtss_f32(folded));
#endif
      }
      return dot;
    }

    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const float d = half_to_float(read_u16_le(block_ptr));
      const float dmin = half_to_float(read_u16_le(block_ptr + sizeof(std::uint16_t)));
      const std::uint8_t *packed_scales = block_ptr + 2 * sizeof(std::uint16_t);
      const std::uint8_t *quants = packed_scales + k_q4_k_scale_bytes;

      int packed_index = 0;
      for (std::uint64_t chunk = 0; chunk < k_qk_k; chunk += 64) {
        std::uint8_t scale0 = 0;
        std::uint8_t min0 = 0;
        std::uint8_t scale1 = 0;
        std::uint8_t min1 = 0;
        get_scale_min_k4(packed_index + 0, packed_scales, &scale0, &min0);
        get_scale_min_k4(packed_index + 1, packed_scales, &scale1, &min1);
        const float d0 = d * static_cast<float>(scale0);
        const float m0 = dmin * static_cast<float>(min0);
        const float d1 = d * static_cast<float>(scale1);
        const float m1 = dmin * static_cast<float>(min1);
        const std::uint8_t *chunk_ptr = quants + (chunk / 2);

        for (std::uint64_t item = 0; item < 32; item += 1) {
          const std::uint64_t low_index = block * k_qk_k + chunk + item;
          const std::uint64_t high_index = low_index + 32;
          const std::uint8_t packed = chunk_ptr[item];
          if (low_index < tensor.element_count) {
            accumulate(low_index, d0 * static_cast<float>(packed & 0x0F) - m0);
          }
          if (high_index < tensor.element_count) {
            accumulate(high_index, d1 * static_cast<float>(packed >> 4) - m1);
          }
        }
        packed_index += 2;
      }
    }
    return dot;
  }

  if (tensor.ggml_type == 13) {
    const auto get_scale_min_k4 = [](int j, const std::uint8_t *packed, std::uint8_t *scale, std::uint8_t *min) {
      if (j < 4) {
        *scale = packed[j] & 63;
        *min = packed[j + 4] & 63;
      } else {
        *scale = static_cast<std::uint8_t>((packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4));
        *min = static_cast<std::uint8_t>((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
      }
    };

    if (aligned_full_blocks) {
      for (std::uint64_t block = first_block; block <= last_block; block += 1) {
        const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
        const float d = half_to_float(read_u16_le(block_ptr));
        const float dmin = half_to_float(read_u16_le(block_ptr + sizeof(std::uint16_t)));
        const std::uint8_t *packed_scales = block_ptr + 2 * sizeof(std::uint16_t);
        const std::uint8_t *qh = packed_scales + k_q4_k_scale_bytes;
        const std::uint8_t *ql = qh + (k_qk_k / 8);
        const std::size_t input_block_offset = static_cast<std::size_t>((block - first_block) * k_qk_k);

        int packed_index = 0;
        std::uint8_t high_bit_low = 1;
        std::uint8_t high_bit_high = 2;
        for (std::uint64_t chunk = 0; chunk < k_qk_k; chunk += 64) {
          std::uint8_t scale0 = 0;
          std::uint8_t min0 = 0;
          std::uint8_t scale1 = 0;
          std::uint8_t min1 = 0;
          get_scale_min_k4(packed_index + 0, packed_scales, &scale0, &min0);
          get_scale_min_k4(packed_index + 1, packed_scales, &scale1, &min1);
          const float d0 = d * static_cast<float>(scale0);
          const float m0 = dmin * static_cast<float>(min0);
          const float d1 = d * static_cast<float>(scale1);
          const float m1 = dmin * static_cast<float>(min1);
          const std::uint8_t *chunk_ptr = ql + (chunk / 2);
          const std::size_t input_chunk_offset = input_block_offset + static_cast<std::size_t>(chunk);

          // Split low/high accumulators; see the Q4_K path for the reasoning
          // and the measurement.
          float low_sum = 0.0f;
          float high_sum = 0.0f;
          for (std::uint64_t item = 0; item < 32; item += 1) {
            const std::uint8_t packed = chunk_ptr[item];
            const std::size_t low_index = input_chunk_offset + static_cast<std::size_t>(item);
            const std::size_t high_index = low_index + 32;
            low_sum += (d0 * static_cast<float>(
              (packed & 0x0F) + ((qh[item] & high_bit_low) != 0 ? 16 : 0)) - m0) * input[low_index];
            high_sum += (d1 * static_cast<float>(
              (packed >> 4) + ((qh[item] & high_bit_high) != 0 ? 16 : 0)) - m1) * input[high_index];
          }
          dot += static_cast<double>(low_sum) + static_cast<double>(high_sum);
          packed_index += 2;
          high_bit_low <<= 2;
          high_bit_high <<= 2;
        }
      }
      return dot;
    }
  }

  if (tensor.ggml_type == 14) {
    if (aligned_full_blocks) {
      // Declared outside the block loop so the four chains stay live across
      // blocks; they are folded into `dot` once, after every block is walked.
      float sum0 = 0.0f;
      float sum1 = 0.0f;
      float sum2 = 0.0f;
      float sum3 = 0.0f;
      for (std::uint64_t block = first_block; block <= last_block; block += 1) {
        const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
        const std::uint8_t *ql = block_ptr;
        const std::uint8_t *qh = ql + (k_qk_k / 2);
        const std::int8_t *scales = reinterpret_cast<const std::int8_t *>(qh + (k_qk_k / 4));
        const float d = half_to_float(read_u16_le(block_ptr + layout->bytes_per_block - sizeof(std::uint16_t)));
        const std::size_t input_block_offset = static_cast<std::size_t>((block - first_block) * k_qk_k);

        for (std::uint64_t chunk = 0; chunk < k_qk_k; chunk += 128) {
          const std::uint8_t *chunk_ql = ql + (chunk / 2);
          const std::uint8_t *chunk_qh = qh + (chunk / 4);
          const std::int8_t *chunk_scales = scales + (chunk / 16);
          const std::size_t input_chunk_offset = input_block_offset + static_cast<std::size_t>(chunk);
          for (std::uint64_t item = 0; item < 32; item += 1) {
            const int scale_index = static_cast<int>(item / 16);
            const std::int8_t q1 = static_cast<std::int8_t>(
              (chunk_ql[item] & 0x0F) | (((chunk_qh[item] >> 0) & 0x03) << 4)) - 32;
            const std::int8_t q2 = static_cast<std::int8_t>(
              (chunk_ql[item + 32] & 0x0F) | (((chunk_qh[item] >> 2) & 0x03) << 4)) - 32;
            const std::int8_t q3 = static_cast<std::int8_t>(
              (chunk_ql[item] >> 4) | (((chunk_qh[item] >> 4) & 0x03) << 4)) - 32;
            const std::int8_t q4 = static_cast<std::int8_t>(
              (chunk_ql[item + 32] >> 4) | (((chunk_qh[item] >> 6) & 0x03) << 4)) - 32;
            const std::size_t index0 = input_chunk_offset + static_cast<std::size_t>(item);
            const std::size_t index1 = index0 + 32;
            const std::size_t index2 = index0 + 64;
            const std::size_t index3 = index0 + 96;
            // Four independent accumulators, one per quarter, folded after the
            // loop. Q6_K unpacks four values per byte-pair, and chaining all
            // four into one f64 total made a dependency chain four times longer
            // than the other formats'. See the Q4_K path for the measurement.
            sum0 += (d * static_cast<float>(chunk_scales[scale_index + 0]) * static_cast<float>(q1)) * input[index0];
            sum1 += (d * static_cast<float>(chunk_scales[scale_index + 2]) * static_cast<float>(q2)) * input[index1];
            sum2 += (d * static_cast<float>(chunk_scales[scale_index + 4]) * static_cast<float>(q3)) * input[index2];
            sum3 += (d * static_cast<float>(chunk_scales[scale_index + 6]) * static_cast<float>(q4)) * input[index3];
          }
        }
      }
      dot += static_cast<double>(sum0) + static_cast<double>(sum1)
        + static_cast<double>(sum2) + static_cast<double>(sum3);
      return dot;
    }
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const std::uint8_t *ql = block_ptr;
      const std::uint8_t *qh = ql + (k_qk_k / 2);
      const std::int8_t *scales = reinterpret_cast<const std::int8_t *>(qh + (k_qk_k / 4));
      const float d = half_to_float(read_u16_le(block_ptr + layout->bytes_per_block - sizeof(std::uint16_t)));

      for (std::uint64_t chunk = 0; chunk < k_qk_k; chunk += 128) {
        const std::uint8_t *chunk_ql = ql + (chunk / 2);
        const std::uint8_t *chunk_qh = qh + (chunk / 4);
        const std::int8_t *chunk_scales = scales + (chunk / 16);
        for (std::uint64_t item = 0; item < 32; item += 1) {
          const int scale_index = static_cast<int>(item / 16);
          const std::int8_t q1 = static_cast<std::int8_t>(
            (chunk_ql[item] & 0x0F) | (((chunk_qh[item] >> 0) & 0x03) << 4)) - 32;
          const std::int8_t q2 = static_cast<std::int8_t>(
            (chunk_ql[item + 32] & 0x0F) | (((chunk_qh[item] >> 2) & 0x03) << 4)) - 32;
          const std::int8_t q3 = static_cast<std::int8_t>(
            (chunk_ql[item] >> 4) | (((chunk_qh[item] >> 4) & 0x03) << 4)) - 32;
          const std::int8_t q4 = static_cast<std::int8_t>(
            (chunk_ql[item + 32] >> 4) | (((chunk_qh[item] >> 6) & 0x03) << 4)) - 32;
          const std::uint64_t index0 = block * k_qk_k + chunk + item;
          const std::uint64_t index1 = index0 + 32;
          const std::uint64_t index2 = index0 + 64;
          const std::uint64_t index3 = index0 + 96;
          if (index0 < tensor.element_count) {
            accumulate(index0, d * static_cast<float>(chunk_scales[scale_index + 0]) * static_cast<float>(q1));
          }
          if (index1 < tensor.element_count) {
            accumulate(index1, d * static_cast<float>(chunk_scales[scale_index + 2]) * static_cast<float>(q2));
          }
          if (index2 < tensor.element_count) {
            accumulate(index2, d * static_cast<float>(chunk_scales[scale_index + 4]) * static_cast<float>(q3));
          }
          if (index3 < tensor.element_count) {
            accumulate(index3, d * static_cast<float>(chunk_scales[scale_index + 6]) * static_cast<float>(q4));
          }
        }
      }
    }
    return dot;
  }

  if (tensor.ggml_type == 10 && aligned_full_blocks) {
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const std::uint8_t *scales = block_ptr;
      const std::uint8_t *quants = block_ptr + (k_qk_k / 16);
      const float d = half_to_float(read_u16_le(block_ptr + 80));
      const float dmin = half_to_float(read_u16_le(block_ptr + 82));
      std::size_t input_position = static_cast<std::size_t>((block - first_block) * k_qk_k);
      int scale_index = 0;
      for (std::uint64_t half = 0; half < k_qk_k; half += 128) {
        const std::uint8_t *q = quants + (half / 4);
        for (int shift = 0; shift < 8; shift += 2) {
          for (int group = 0; group < 2; group += 1) {
            const std::uint8_t packed_scale = scales[scale_index];
            scale_index += 1;
            const float dl = d * static_cast<float>(packed_scale & 0x0F);
            const float ml = dmin * static_cast<float>(packed_scale >> 4);
            for (int item = 0; item < 16; item += 1) {
              dot += static_cast<double>(dl * static_cast<float>((q[item + group * 16] >> shift) & 3) - ml)
                * static_cast<double>(input[input_position]);
              input_position += 1;
            }
          }
        }
      }
    }
    return dot;
  }

  if (tensor.ggml_type == 11 && aligned_full_blocks) {
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const std::uint8_t *hmask = block_ptr;
      const std::uint8_t *quants = block_ptr + (k_qk_k / 8);
      const std::uint8_t *packed_scales = block_ptr + (k_qk_k / 8) + (k_qk_k / 4);
      const float d = half_to_float(read_u16_le(block_ptr + layout->bytes_per_block - sizeof(std::uint16_t)));
      int sub_scales[16];
      for (int j = 0; j < 16; j += 1) {
        const int low4 = j < 8
          ? (packed_scales[j] & 0x0F)
          : (packed_scales[j - 8] >> 4);
        const int high2 = (packed_scales[8 + (j % 4)] >> (2 * (j / 4))) & 0x03;
        sub_scales[j] = low4 | (high2 << 4);
      }

      std::size_t input_position = static_cast<std::size_t>((block - first_block) * k_qk_k);
      int scale_index = 0;
      std::uint8_t mask_bit = 1;
      for (std::uint64_t half = 0; half < k_qk_k; half += 128) {
        const std::uint8_t *q = quants + (half / 4);
        for (int shift = 0; shift < 8; shift += 2) {
          for (int group = 0; group < 2; group += 1) {
            const float dl = d * static_cast<float>(sub_scales[scale_index] - 32);
            scale_index += 1;
            for (int item = 0; item < 16; item += 1) {
              const int source = item + group * 16;
              int q_value = (q[source] >> shift) & 3;
              if ((hmask[source] & mask_bit) == 0) {
                q_value -= 4;
              }
              dot += static_cast<double>(dl * static_cast<float>(q_value))
                * static_cast<double>(input[input_position]);
              input_position += 1;
            }
          }
          mask_bit = static_cast<std::uint8_t>(mask_bit << 1);
        }
      }
    }
    return dot;
  }

  if (tensor.ggml_type == 23 && aligned_full_blocks) {
    static const std::int8_t k_iq4_nl_values[16] = {
      -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const float d = half_to_float(read_u16_le(block_ptr));
      const std::uint32_t scales_h = read_u16_le(block_ptr + 2);
      const std::uint8_t *scales_l = block_ptr + 4;
      const std::uint8_t *quants = block_ptr + 8;
      std::size_t input_block_offset = static_cast<std::size_t>((block - first_block) * k_qk_k);
      for (int sub = 0; sub < 8; sub += 1) {
        const int ls = ((scales_l[sub / 2] >> (4 * (sub % 2))) & 0x0F)
          | (static_cast<int>((scales_h >> (2 * sub)) & 0x03) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        const std::uint8_t *q = quants + sub * 16;
        const std::size_t sub_offset = input_block_offset + static_cast<std::size_t>(sub) * 32;
        for (int item = 0; item < 16; item += 1) {
          dot += static_cast<double>(dl * static_cast<float>(k_iq4_nl_values[q[item] & 0x0F]))
            * static_cast<double>(input[sub_offset + static_cast<std::size_t>(item)]);
          dot += static_cast<double>(dl * static_cast<float>(k_iq4_nl_values[q[item] >> 4]))
            * static_cast<double>(input[sub_offset + static_cast<std::size_t>(item) + 16]);
        }
      }
    }
    return dot;
  }

  if (tensor.ggml_type == 34 && aligned_full_blocks) {
    static const std::uint8_t k_pow3[6] = {1, 3, 9, 27, 81, 243};
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const std::uint8_t *qs = block_ptr;
      const std::uint8_t *qh = block_ptr + 48;
      const float d = half_to_float(read_u16_le(block_ptr + 52));
      std::size_t input_position = static_cast<std::size_t>((block - first_block) * k_qk_k);
      const auto emit_trit = [&](std::uint8_t packed, int digit) {
        const std::uint8_t scaled = static_cast<std::uint8_t>(packed * k_pow3[digit]);
        const int trit = static_cast<int>((static_cast<std::uint16_t>(scaled) * 3) >> 8) - 1;
        dot += static_cast<double>(static_cast<float>(trit) * d)
          * static_cast<double>(input[input_position]);
        input_position += 1;
      };
      for (int digit = 0; digit < 5; digit += 1) {
        for (int item = 0; item < 32; item += 1) {
          emit_trit(qs[item], digit);
        }
      }
      for (int digit = 0; digit < 5; digit += 1) {
        for (int item = 0; item < 16; item += 1) {
          emit_trit(qs[32 + item], digit);
        }
      }
      for (int digit = 0; digit < 4; digit += 1) {
        for (int item = 0; item < 4; item += 1) {
          emit_trit(qh[item], digit);
        }
      }
    }
    return dot;
  }

  if (tensor.ggml_type == 35 && aligned_full_blocks) {
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *block_ptr = data + block * layout->bytes_per_block;
      const std::uint8_t *qs = block_ptr;
      const float d = half_to_float(read_u16_le(block_ptr + 64));
      std::size_t input_position = static_cast<std::size_t>((block - first_block) * k_qk_k);
      for (int chunk = 0; chunk < 64; chunk += 32) {
        for (int shift = 0; shift < 8; shift += 2) {
          for (int item = 0; item < 32; item += 1) {
            const int q_value = static_cast<int>((qs[chunk + item] >> shift) & 3) - 1;
            dot += static_cast<double>(static_cast<float>(q_value) * d)
              * static_cast<double>(input[input_position]);
            input_position += 1;
          }
        }
      }
    }
    return dot;
  }

  // Rare partial-range calls still fall back to shared dequantization. Decode
  // row matvecs hit the direct accumulation paths above when rows are block-aligned.
  std::vector<float> values;
  dequantize_tensor_range_into_f32(state, tensor, start, count, values);
  for (std::size_t index = 0; index < values.size(); index += 1) {
    dot += static_cast<double>(values[index]) * static_cast<double>(input[index]);
  }
  return dot;
}

// CPU decode for CPU-only quantizations spends most of its time in the FFN
// gate/up pair. When both tensors share a quant format, walk the packed rows
// once and accumulate both dots together so the input stream and block metadata
// are reused.
static bool tensor_range_dual_dot_f32(
  const LoadedModelState &state,
  const TensorInfo &left_tensor,
  const TensorInfo &right_tensor,
  std::uint64_t start,
  std::uint64_t count,
  const std::vector<float> &input,
  double *left_dot,
  double *right_dot) {
  if (left_dot == nullptr
    || right_dot == nullptr
    || left_tensor.ggml_type != right_tensor.ggml_type
    || count != input.size()) {
    return false;
  }
  const auto layout = ggml_type_layout(left_tensor.ggml_type);
  if (!layout.has_value()) {
    return false;
  }
  const std::uint64_t end = start + count;
  const bool aligned_full_blocks = (start % layout->block_size) == 0 && (count % layout->block_size) == 0;
  if (!aligned_full_blocks) {
    return false;
  }
  const std::uint64_t first_block = start / layout->block_size;
  const std::uint64_t last_block = (end - 1) / layout->block_size;
  const std::uint8_t *left_data = tensor_data_ptr(state, left_tensor);
  const std::uint8_t *right_data = tensor_data_ptr(state, right_tensor);
  double left = 0.0;
  double right = 0.0;

  if (left_tensor.ggml_type == 10) {
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *left_block_ptr = left_data + block * layout->bytes_per_block;
      const std::uint8_t *right_block_ptr = right_data + block * layout->bytes_per_block;
      const std::uint8_t *left_scales = left_block_ptr;
      const std::uint8_t *right_scales = right_block_ptr;
      const std::uint8_t *left_quants = left_block_ptr + (k_qk_k / 16);
      const std::uint8_t *right_quants = right_block_ptr + (k_qk_k / 16);
      const float left_d = half_to_float(read_u16_le(left_block_ptr + 80));
      const float left_dmin = half_to_float(read_u16_le(left_block_ptr + 82));
      const float right_d = half_to_float(read_u16_le(right_block_ptr + 80));
      const float right_dmin = half_to_float(read_u16_le(right_block_ptr + 82));
      std::size_t input_position = static_cast<std::size_t>((block - first_block) * k_qk_k);
      int scale_index = 0;
      for (std::uint64_t half = 0; half < k_qk_k; half += 128) {
        const std::uint8_t *left_q = left_quants + (half / 4);
        const std::uint8_t *right_q = right_quants + (half / 4);
        for (int shift = 0; shift < 8; shift += 2) {
          for (int group = 0; group < 2; group += 1) {
            const std::uint8_t left_packed_scale = left_scales[scale_index];
            const std::uint8_t right_packed_scale = right_scales[scale_index];
            scale_index += 1;
            const float left_dl = left_d * static_cast<float>(left_packed_scale & 0x0F);
            const float left_ml = left_dmin * static_cast<float>(left_packed_scale >> 4);
            const float right_dl = right_d * static_cast<float>(right_packed_scale & 0x0F);
            const float right_ml = right_dmin * static_cast<float>(right_packed_scale >> 4);
            for (int item = 0; item < 16; item += 1) {
              const int source = item + group * 16;
              const float input_value = input[input_position];
              left += static_cast<double>(left_dl * static_cast<float>((left_q[source] >> shift) & 3) - left_ml)
                * static_cast<double>(input_value);
              right += static_cast<double>(right_dl * static_cast<float>((right_q[source] >> shift) & 3) - right_ml)
                * static_cast<double>(input_value);
              input_position += 1;
            }
          }
        }
      }
    }
    *left_dot = left;
    *right_dot = right;
    return true;
  }

  if (left_tensor.ggml_type == 2) {
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *left_block_ptr = left_data + block * layout->bytes_per_block;
      const std::uint8_t *right_block_ptr = right_data + block * layout->bytes_per_block;
      const float left_d = half_to_float(read_u16_le(left_block_ptr));
      const float right_d = half_to_float(read_u16_le(right_block_ptr));
      const std::uint8_t *left_quants = left_block_ptr + sizeof(std::uint16_t);
      const std::uint8_t *right_quants = right_block_ptr + sizeof(std::uint16_t);
      const std::size_t input_offset = static_cast<std::size_t>((block - first_block) * layout->block_size);
      for (std::uint64_t item = 0; item < 16; item += 1) {
        const float input_low = input[input_offset + static_cast<std::size_t>(item)];
        const float input_high = input[input_offset + static_cast<std::size_t>(item) + 16];
        const std::uint8_t left_packed = left_quants[item];
        const std::uint8_t right_packed = right_quants[item];
        left += static_cast<double>(left_d * (static_cast<float>(left_packed & 0x0F) - 8.0f))
          * static_cast<double>(input_low);
        left += static_cast<double>(left_d * (static_cast<float>(left_packed >> 4) - 8.0f))
          * static_cast<double>(input_high);
        right += static_cast<double>(right_d * (static_cast<float>(right_packed & 0x0F) - 8.0f))
          * static_cast<double>(input_low);
        right += static_cast<double>(right_d * (static_cast<float>(right_packed >> 4) - 8.0f))
          * static_cast<double>(input_high);
      }
    }
    *left_dot = left;
    *right_dot = right;
    return true;
  }

  if (left_tensor.ggml_type == 35) {
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *left_block_ptr = left_data + block * layout->bytes_per_block;
      const std::uint8_t *right_block_ptr = right_data + block * layout->bytes_per_block;
      const std::uint8_t *left_qs = left_block_ptr;
      const std::uint8_t *right_qs = right_block_ptr;
      const float left_d = half_to_float(read_u16_le(left_block_ptr + 64));
      const float right_d = half_to_float(read_u16_le(right_block_ptr + 64));
      std::size_t input_position = static_cast<std::size_t>((block - first_block) * k_qk_k);
      for (int chunk = 0; chunk < 64; chunk += 32) {
        for (int shift = 0; shift < 8; shift += 2) {
          for (int item = 0; item < 32; item += 1) {
            const float input_value = input[input_position];
            const int left_q_value = static_cast<int>((left_qs[chunk + item] >> shift) & 3) - 1;
            const int right_q_value = static_cast<int>((right_qs[chunk + item] >> shift) & 3) - 1;
            left += static_cast<double>(static_cast<float>(left_q_value) * left_d)
              * static_cast<double>(input_value);
            right += static_cast<double>(static_cast<float>(right_q_value) * right_d)
              * static_cast<double>(input_value);
            input_position += 1;
          }
        }
      }
    }
    *left_dot = left;
    *right_dot = right;
    return true;
  }

  if (left_tensor.ggml_type == 34) {
    static const std::uint8_t k_pow3[6] = {1, 3, 9, 27, 81, 243};
    for (std::uint64_t block = first_block; block <= last_block; block += 1) {
      const std::uint8_t *left_block_ptr = left_data + block * layout->bytes_per_block;
      const std::uint8_t *right_block_ptr = right_data + block * layout->bytes_per_block;
      const std::uint8_t *left_qs = left_block_ptr;
      const std::uint8_t *right_qs = right_block_ptr;
      const std::uint8_t *left_qh = left_block_ptr + 48;
      const std::uint8_t *right_qh = right_block_ptr + 48;
      const float left_d = half_to_float(read_u16_le(left_block_ptr + 52));
      const float right_d = half_to_float(read_u16_le(right_block_ptr + 52));
      std::size_t input_position = static_cast<std::size_t>((block - first_block) * k_qk_k);
      const auto dual_emit_trit = [&](std::uint8_t left_packed, std::uint8_t right_packed, int digit) {
        const std::uint8_t left_scaled = static_cast<std::uint8_t>(left_packed * k_pow3[digit]);
        const std::uint8_t right_scaled = static_cast<std::uint8_t>(right_packed * k_pow3[digit]);
        const int left_trit = static_cast<int>((static_cast<std::uint16_t>(left_scaled) * 3) >> 8) - 1;
        const int right_trit = static_cast<int>((static_cast<std::uint16_t>(right_scaled) * 3) >> 8) - 1;
        const float input_value = input[input_position];
        left += static_cast<double>(static_cast<float>(left_trit) * left_d)
          * static_cast<double>(input_value);
        right += static_cast<double>(static_cast<float>(right_trit) * right_d)
          * static_cast<double>(input_value);
        input_position += 1;
      };
      for (int digit = 0; digit < 5; digit += 1) {
        for (int item = 0; item < 32; item += 1) {
          dual_emit_trit(left_qs[item], right_qs[item], digit);
        }
      }
      for (int digit = 0; digit < 5; digit += 1) {
        for (int item = 0; item < 16; item += 1) {
          dual_emit_trit(left_qs[32 + item], right_qs[32 + item], digit);
        }
      }
      for (int digit = 0; digit < 4; digit += 1) {
        for (int item = 0; item < 4; item += 1) {
          dual_emit_trit(left_qh[item], right_qh[item], digit);
        }
      }
    }
    *left_dot = left;
    *right_dot = right;
    return true;
  }

  return false;
}

static unsigned int max_cpu_row_workers() {
  static const unsigned int value = [] {
    const unsigned int detected = std::max(1u, std::thread::hardware_concurrency());
    const char *configured = std::getenv("LIZARD_NATIVE_CPU_WORKERS");
    if (configured != nullptr && configured[0] != '\0') {
      const unsigned long parsed = std::strtoul(configured, nullptr, 10);
      if (parsed > 0) {
        return std::max(1u, std::min<unsigned int>(detected, static_cast<unsigned int>(parsed)));
      }
    }
    // Decode-heavy CPU paths on this workstation still have spare cores with
    // the previous 8-worker cap. Keep some headroom for the UI/server while
    // letting larger FFN and matvec jobs spread wider.
    return std::max(1u, std::min(12u, detected));
  }();
  return value;
}

class PersistentRowWorkers {
public:
  PersistentRowWorkers()
    : thread_count_(max_cpu_row_workers()) {
    workers_.reserve(thread_count_);
    for (unsigned int worker = 0; worker < thread_count_; worker += 1) {
      workers_.emplace_back([this, worker]() { run(worker); });
    }
  }

  ~PersistentRowWorkers() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      generation_ += 1;
    }
    task_ready_.notify_all();
    for (std::thread &worker : workers_) {
      worker.join();
    }
  }

  void execute(
    std::uint64_t rows,
    std::uint64_t worker_count,
    const std::function<void(std::uint64_t, std::uint64_t)> &task) {
    const std::uint64_t active_workers = std::min<std::uint64_t>(worker_count, thread_count_);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      rows_ = rows;
      active_workers_ = active_workers;
      completed_workers_ = 0;
      task_ = task;
      generation_ += 1;
    }
    task_ready_.notify_all();
    std::unique_lock<std::mutex> lock(mutex_);
    task_done_.wait(lock, [&]() { return completed_workers_ == active_workers; });
    task_ = {};
  }

private:
  void run(unsigned int worker_index) {
    std::uint64_t observed_generation = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      task_ready_.wait(lock, [&]() {
        return stopping_ || generation_ != observed_generation;
      });
      if (stopping_) {
        return;
      }
      observed_generation = generation_;
      if (worker_index >= active_workers_) {
        continue;
      }
      const std::uint64_t rows = rows_;
      const std::uint64_t active_workers = active_workers_;
      const auto task = task_;
      lock.unlock();

      const std::uint64_t start_row = (rows * worker_index) / active_workers;
      const std::uint64_t end_row = (rows * (worker_index + 1)) / active_workers;
      task(start_row, end_row);

      lock.lock();
      completed_workers_ += 1;
      if (completed_workers_ == active_workers_) {
        task_done_.notify_one();
      }
    }
  }

  unsigned int thread_count_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable task_ready_;
  std::condition_variable task_done_;
  bool stopping_ = false;
  std::uint64_t generation_ = 0;
  std::uint64_t rows_ = 0;
  std::uint64_t active_workers_ = 0;
  std::uint64_t completed_workers_ = 0;
  std::function<void(std::uint64_t, std::uint64_t)> task_;
};

static PersistentRowWorkers &persistent_row_workers() {
  static PersistentRowWorkers workers;
  return workers;
}

#ifdef LIZARD_NATIVE_AVX2
// Q6_K weights are designed to be dotted with one Q8_K activation block. The
// generic path above expands every packed weight to float for every output row;
// on the tied 128256 x 3072 Llama vocabulary projection that held the LM head
// to roughly 1.3 GB/s. Quantizing the 3072-value activation once lets every row
// use AVX2 integer dot products while the 323 MB packed matrix is streamed only
// once. This follows GGML's Q6_K x Q8_K arithmetic, but stays local so the
// independent runner does not acquire a llama.cpp runtime dependency.
struct Q8KActivationBlock {
  float delta = 0.0f;
  alignas(32) std::int8_t quants[k_qk_k] {};
  alignas(32) std::int16_t sums[k_qk_k / 16] {};
};

static bool q6_k_q8_k_fastpath_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_Q6_Q8_FASTPATH");
    return value == nullptr || std::string(value) != "0";
  }();
  return enabled;
}

static std::vector<Q8KActivationBlock> quantize_q8_k_activation(
  const std::vector<float> &input) {
  if (input.empty() || input.size() % k_qk_k != 0) {
    return {};
  }
  std::vector<Q8KActivationBlock> blocks(input.size() / k_qk_k);
  for (std::size_t block_index = 0; block_index < blocks.size(); block_index += 1) {
    Q8KActivationBlock &block = blocks[block_index];
    const float *values = input.data() + block_index * k_qk_k;
    float signed_max = 0.0f;
    float absolute_max = 0.0f;
    for (std::size_t item = 0; item < k_qk_k; item += 1) {
      const float magnitude = std::abs(values[item]);
      if (magnitude > absolute_max) {
        absolute_max = magnitude;
        signed_max = values[item];
      }
    }
    if (absolute_max == 0.0f) {
      continue;
    }
    // Preserve GGML Q8_K's signed-scale convention. Choosing the sign of the
    // largest-magnitude value keeps every rounded quant in [-127, 127].
    const float inverse_delta = -127.0f / signed_max;
    block.delta = 1.0f / inverse_delta;
    for (std::size_t group = 0; group < k_qk_k / 16; group += 1) {
      int sum = 0;
      for (std::size_t lane = 0; lane < 16; lane += 1) {
        const std::size_t item = group * 16 + lane;
        const int quant = std::min(127, static_cast<int>(std::nearbyint(inverse_delta * values[item])));
        block.quants[item] = static_cast<std::int8_t>(quant);
        sum += quant;
      }
      block.sums[group] = static_cast<std::int16_t>(sum);
    }
  }
  return blocks;
}

static std::int32_t horizontal_sum_i32(__m256i values) {
  const __m128i low = _mm256_castsi256_si128(values);
  const __m128i high = _mm256_extracti128_si256(values, 1);
  __m128i sum = _mm_add_epi32(low, high);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

static float horizontal_sum_f32(__m256 values) {
  __m128 sum = _mm_add_ps(
    _mm256_castps256_ps128(values),
    _mm256_extractf128_ps(values, 1));
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
  return _mm_cvtss_f32(sum);
}

static float q6_k_q8_k_row_dot_avx2(
  const std::uint8_t *row_data,
  const std::vector<Q8KActivationBlock> &activations) {
  const __m256i low_two_bits = _mm256_set1_epi8(3);
  const __m256i low_nibble = _mm256_set1_epi8(15);
  __m256 vector_total = _mm256_setzero_ps();
  double correction_total = 0.0;
  for (std::size_t block_index = 0; block_index < activations.size(); block_index += 1) {
    const std::uint8_t *block = row_data + block_index * 210u;
    const std::uint8_t *low_quants = block;
    const std::uint8_t *high_quants = low_quants + (k_qk_k / 2);
    const std::int8_t *scales = reinterpret_cast<const std::int8_t *>(high_quants + (k_qk_k / 4));
    const float weight_delta = half_to_float(read_u16_le(block + 208));
    const Q8KActivationBlock &activation = activations[block_index];
    __m256i accumulated = _mm256_setzero_si256();

    for (int half = 0; half < 2; half += 1) {
      const std::uint8_t *low = low_quants + half * 64;
      const __m256i low_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(low));
      const __m256i low_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(low + 32));
      const __m256i high = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(high_quants + half * 32));
      const __m256i unpacked[4] = {
        _mm256_or_si256(
          _mm256_and_si256(low_0, low_nibble),
          _mm256_slli_epi16(_mm256_and_si256(high, low_two_bits), 4)),
        _mm256_or_si256(
          _mm256_and_si256(low_1, low_nibble),
          _mm256_slli_epi16(_mm256_and_si256(high, _mm256_set1_epi8(12)), 2)),
        _mm256_or_si256(
          _mm256_and_si256(_mm256_srli_epi16(low_0, 4), low_nibble),
          _mm256_and_si256(high, _mm256_set1_epi8(48))),
        _mm256_or_si256(
          _mm256_and_si256(_mm256_srli_epi16(low_1, 4), low_nibble),
          _mm256_srli_epi16(_mm256_and_si256(high, _mm256_set1_epi8(-64)), 2)),
      };

      for (int quarter = 0; quarter < 4; quarter += 1) {
        const int scale_index = half * 8 + quarter * 2;
        const __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
          activation.quants + half * 128 + quarter * 32));
        const __m256i pair_dots = _mm256_maddubs_epi16(unpacked[quarter], q8);
        const __m256i scale_vector = _mm256_set_m128i(
          _mm_set1_epi16(scales[scale_index + 1]),
          _mm_set1_epi16(scales[scale_index]));
        accumulated = _mm256_add_epi32(
          accumulated,
          _mm256_madd_epi16(pair_dots, scale_vector));
      }
    }

    const __m256i scale_values = _mm256_cvtepi8_epi16(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(scales)));
    const __m256i activation_sums = _mm256_load_si256(
      reinterpret_cast<const __m256i *>(activation.sums));
    const std::int32_t offset_correction = horizontal_sum_i32(
      _mm256_madd_epi16(scale_values, activation_sums));
    const float scale = weight_delta * activation.delta;
    vector_total = _mm256_fmadd_ps(
      _mm256_set1_ps(scale),
      _mm256_cvtepi32_ps(accumulated),
      vector_total);
    correction_total -= static_cast<double>(scale)
      * static_cast<double>(32 * offset_correction);
  }
  return horizontal_sum_f32(vector_total) + static_cast<float>(correction_total);
}

static bool q6_k_q8_k_kernel_selftest() {
  std::vector<float> input(k_qk_k, 0.0f);
  for (std::size_t item = 0; item < input.size(); item += 1) {
    input[item] = static_cast<float>((static_cast<int>(item % 29) - 14) * 0.03125);
  }
  const std::vector<Q8KActivationBlock> activations = quantize_q8_k_activation(input);
  if (activations.size() != 1) {
    return false;
  }

  alignas(32) std::uint8_t row[210] {};
  for (std::size_t item = 0; item < k_qk_k / 2; item += 1) {
    row[item] = static_cast<std::uint8_t>((item * 37u + 11u) & 0xFFu);
  }
  for (std::size_t item = 0; item < k_qk_k / 4; item += 1) {
    row[k_qk_k / 2 + item] = static_cast<std::uint8_t>((item * 19u + 7u) & 0xFFu);
  }
  std::int8_t *scales = reinterpret_cast<std::int8_t *>(row + 192);
  for (int group = 0; group < static_cast<int>(k_qk_k / 16); group += 1) {
    scales[group] = static_cast<std::int8_t>((group * 7) % 23 - 11);
  }
  // IEEE-754 half 1.0, little endian.
  row[208] = 0x00;
  row[209] = 0x3C;

  std::int64_t reference_integer_dot = 0;
  for (int half = 0; half < 2; half += 1) {
    const std::uint8_t *low = row + half * 64;
    const std::uint8_t *high = row + 128 + half * 32;
    for (int item = 0; item < 32; item += 1) {
      const int q0 = static_cast<int>((low[item] & 0x0F) | ((high[item] & 0x03) << 4)) - 32;
      const int q1 = static_cast<int>((low[item + 32] & 0x0F) | (((high[item] >> 2) & 0x03) << 4)) - 32;
      const int q2 = static_cast<int>((low[item] >> 4) | (((high[item] >> 4) & 0x03) << 4)) - 32;
      const int q3 = static_cast<int>((low[item + 32] >> 4) | (((high[item] >> 6) & 0x03) << 4)) - 32;
      const int scale_base = half * 8 + item / 16;
      reference_integer_dot += static_cast<std::int64_t>(scales[scale_base + 0]) * q0
        * activations[0].quants[half * 128 + item + 0];
      reference_integer_dot += static_cast<std::int64_t>(scales[scale_base + 2]) * q1
        * activations[0].quants[half * 128 + item + 32];
      reference_integer_dot += static_cast<std::int64_t>(scales[scale_base + 4]) * q2
        * activations[0].quants[half * 128 + item + 64];
      reference_integer_dot += static_cast<std::int64_t>(scales[scale_base + 6]) * q3
        * activations[0].quants[half * 128 + item + 96];
    }
  }
  const float expected = activations[0].delta * static_cast<float>(reference_integer_dot);
  const float actual = q6_k_q8_k_row_dot_avx2(row, activations);
  return std::isfinite(actual)
    && std::abs(actual - expected) <= std::max(1e-4f, std::abs(expected) * 1e-5f);
}

static bool q4_k_q8_k_fastpath_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_Q4_Q8_FASTPATH");
    return value == nullptr || std::string(value) != "0";
  }();
  return enabled;
}

static void decode_q4_k_scale_min(
  int index,
  const std::uint8_t *packed,
  std::uint8_t *scale,
  std::uint8_t *minimum) {
  if (index < 4) {
    *scale = packed[index] & 63;
    *minimum = packed[index + 4] & 63;
  } else {
    *scale = static_cast<std::uint8_t>((packed[index + 4] & 0x0F) | ((packed[index - 4] >> 6) << 4));
    *minimum = static_cast<std::uint8_t>((packed[index + 4] >> 4) | ((packed[index] >> 6) << 4));
  }
}

static float q4_k_q8_k_row_dot_avx2(
  const std::uint8_t *row_data,
  const std::vector<Q8KActivationBlock> &activations) {
  const __m256i low_nibble = _mm256_set1_epi8(15);
  __m256 vector_total = _mm256_setzero_ps();
  double minimum_total = 0.0;
  for (std::size_t block_index = 0; block_index < activations.size(); block_index += 1) {
    const std::uint8_t *block = row_data + block_index * 144u;
    const float weight_delta = half_to_float(read_u16_le(block));
    const float minimum_delta = half_to_float(read_u16_le(block + 2));
    const std::uint8_t *packed_scales = block + 4;
    const std::uint8_t *packed_quants = packed_scales + k_q4_k_scale_bytes;
    const Q8KActivationBlock &activation = activations[block_index];
    __m256i accumulated = _mm256_setzero_si256();
    std::uint8_t scales[8] {};
    std::uint8_t minimums[8] {};
    for (int index = 0; index < 8; index += 1) {
      decode_q4_k_scale_min(index, packed_scales, &scales[index], &minimums[index]);
    }

    for (int chunk = 0; chunk < static_cast<int>(k_qk_k); chunk += 64) {
      const int scale_index = chunk / 32;
      const std::uint8_t scale_low = scales[scale_index];
      const std::uint8_t scale_high = scales[scale_index + 1];

      const __m256i packed = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(packed_quants + chunk / 2));
      const __m256i low = _mm256_and_si256(packed, low_nibble);
      const __m256i high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_nibble);
      const __m256i q8_low = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(activation.quants + chunk));
      const __m256i q8_high = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(activation.quants + chunk + 32));
      const __m256i low_pairs = _mm256_maddubs_epi16(low, q8_low);
      const __m256i high_pairs = _mm256_maddubs_epi16(high, q8_high);
      accumulated = _mm256_add_epi32(accumulated, _mm256_madd_epi16(
        low_pairs, _mm256_set1_epi16(static_cast<std::int16_t>(scale_low))));
      accumulated = _mm256_add_epi32(accumulated, _mm256_madd_epi16(
        high_pairs, _mm256_set1_epi16(static_cast<std::int16_t>(scale_high))));

    }

    const __m256i activation_sums = _mm256_load_si256(
      reinterpret_cast<const __m256i *>(activation.sums));
    const __m128i paired_sums = _mm_hadd_epi16(
      _mm256_castsi256_si128(activation_sums),
      _mm256_extracti128_si256(activation_sums, 1));
    const __m128i minimum_values = _mm_cvtepu8_epi16(
      _mm_loadl_epi64(reinterpret_cast<const __m128i *>(minimums)));
    const __m128i minimum_products = _mm_madd_epi16(minimum_values, paired_sums);
    const std::int32_t minimum_correction = horizontal_sum_i32(
      _mm256_inserti128_si256(_mm256_setzero_si256(), minimum_products, 0));

    const float scale = weight_delta * activation.delta;
    vector_total = _mm256_fmadd_ps(
      _mm256_set1_ps(scale),
      _mm256_cvtepi32_ps(accumulated),
      vector_total);
    minimum_total -= static_cast<double>(minimum_delta * activation.delta)
      * static_cast<double>(minimum_correction);
  }
  return horizontal_sum_f32(vector_total) + static_cast<float>(minimum_total);
}

static bool q4_k_q8_k_kernel_selftest() {
  std::vector<float> input(k_qk_k, 0.0f);
  for (std::size_t item = 0; item < input.size(); item += 1) {
    input[item] = static_cast<float>((static_cast<int>(item % 31) - 15) * 0.02734375);
  }
  const std::vector<Q8KActivationBlock> activations = quantize_q8_k_activation(input);
  if (activations.size() != 1) {
    return false;
  }

  alignas(32) std::uint8_t row[144] {};
  row[0] = 0x00;
  row[1] = 0x3C;
  row[2] = 0x00;
  row[3] = 0x38; // half 0.5
  for (int item = 0; item < k_q4_k_scale_bytes; item += 1) {
    row[4 + item] = static_cast<std::uint8_t>((item * 29 + 13) & 0xFF);
  }
  for (int item = 0; item < static_cast<int>(k_qk_k / 2); item += 1) {
    row[16 + item] = static_cast<std::uint8_t>((item * 41 + 5) & 0xFF);
  }

  std::int64_t scaled_dot = 0;
  std::int64_t minimum_dot = 0;
  int packed_index = 0;
  for (int chunk = 0; chunk < static_cast<int>(k_qk_k); chunk += 64) {
    std::uint8_t scale_low = 0;
    std::uint8_t minimum_low = 0;
    std::uint8_t scale_high = 0;
    std::uint8_t minimum_high = 0;
    decode_q4_k_scale_min(packed_index + 0, row + 4, &scale_low, &minimum_low);
    decode_q4_k_scale_min(packed_index + 1, row + 4, &scale_high, &minimum_high);
    packed_index += 2;
    const std::uint8_t *quants = row + 16 + chunk / 2;
    for (int item = 0; item < 32; item += 1) {
      const int q8_low = activations[0].quants[chunk + item];
      const int q8_high = activations[0].quants[chunk + item + 32];
      scaled_dot += static_cast<std::int64_t>(scale_low) * (quants[item] & 0x0F) * q8_low;
      scaled_dot += static_cast<std::int64_t>(scale_high) * (quants[item] >> 4) * q8_high;
      minimum_dot += static_cast<std::int64_t>(minimum_low) * q8_low;
      minimum_dot += static_cast<std::int64_t>(minimum_high) * q8_high;
    }
  }
  const float expected = activations[0].delta
    * (static_cast<float>(scaled_dot) - 0.5f * static_cast<float>(minimum_dot));
  const float actual = q4_k_q8_k_row_dot_avx2(row, activations);
  if (!std::isfinite(actual)
    || std::abs(actual - expected) > std::max(1e-4f, std::abs(expected) * 1e-5f)) {
    return false;
  }
  if (ggml_cpu_backend_compiled()) {
    std::vector<float> ggml_output;
    std::string ggml_error;
    if (!ggml_cpu_backend_matvec(
        12,
        1,
        k_qk_k,
        row,
        sizeof(row),
        input.data(),
        input.size(),
        1,
        &ggml_output,
        &ggml_error)
      || ggml_output.size() != 1
      || !std::isfinite(ggml_output[0])
      || std::abs(ggml_output[0] - expected) > std::max(1e-3f, std::abs(expected) * 2e-4f)) {
      return false;
    }
  }
  return true;
}
#endif

// Whether the current session's execution plan and device allow GPU decode.
// This is independent of the prompt-size threshold used for GPU prefill.
static std::atomic<bool> g_gpu_lane_session_enabled {false};
static thread_local bool g_gpu_decode_required = false;

static bool gpu_decode_lane_requested() {
  static const bool env_enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_GPU_DECODE");
    return value == nullptr || std::string(value) != "0";
  }();
  return env_enabled && g_gpu_lane_session_enabled.load(std::memory_order_relaxed);
}

static std::string gpu_decode_disabled_detail() {
  const char *reason = std::getenv("LIZARD_NATIVE_GPU_DECODE_REASON");
  if (reason != nullptr && reason[0] != '\0') {
    return reason;
  }
  const char *enabled = std::getenv("LIZARD_NATIVE_GPU_DECODE");
  if (enabled != nullptr && std::string(enabled) == "0") {
    return "GPU decode was disabled by the native execution plan; CPU decode remains active.";
  }
  return "";
}

static bool gpu_decode_model_quantization_eligible(const LoadedModelState &state) {
  const std::string quantization = to_lower(state.model.quantization);
  return quantization == "f32"
    || quantization == "f16"
    || quantization == "bf16"
    || quantization == "q8_0"
    || quantization == "q4_0"
    || quantization.rfind("q4_k", 0) == 0
    || quantization.rfind("q5_k", 0) == 0
    || quantization.rfind("q6_k", 0) == 0;
}

static bool model_has_gpu_projection_candidate(const LoadedModelState &state) {
  if (!gpu_decode_model_quantization_eligible(state)) {
    return false;
  }
  for (const TensorInfo &tensor : state.model.tensors) {
    if (tensor.dimensions.size() != 2 || tensor.element_count < 1048576) {
      continue;
    }
    D3d12WeightFormat format = D3d12WeightFormat::F32;
    if (d3d12_weight_format_supported(tensor.ggml_type, tensor.dimensions[0], &format)) {
      return true;
    }
  }
  return false;
}

static bool gpu_decode_lane_enabled_for_model(const LoadedModelState &state) {
  return gpu_decode_lane_requested() && gpu_decode_model_quantization_eligible(state);
}

[[noreturn]] static void throw_required_gpu_decode_failure(
  const std::string &operation,
  const std::string &detail) {
  throw WorkerError(
    "NATIVE_GPU_DECODE_REQUIRED",
    operation + " could not execute on the required Direct3D 12 decode lane"
      + (detail.empty() ? "." : ": " + detail));
}

static std::string gpu_decode_quantization_gate_detail(const LoadedModelState &state) {
  const std::string quantization = state.model.quantization.empty()
    ? "unknown"
    : state.model.quantization;
  return "Decode stayed on CPU because model quantization "
    + quantization
    + " is outside the native resident GPU decode subset (F32/F16/BF16/Q4_0/Q4_K/Q5_K/Q6_K/Q8_0).";
}

static bool gpu_fuse_qkv_enabled() {
  static const bool env_enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_GPU_FUSE_QKV");
    return value == nullptr || std::string(value) != "0";
  }();
  return env_enabled;
}

static bool gpu_fuse_ffn_enabled() {
  static const bool env_enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_GPU_FUSE_FFN");
    return value == nullptr || std::string(value) != "0";
  }();
  return env_enabled;
}

static bool gpu_fuse_layer_enabled() {
  // Refreshed 2026-07-04 on Iris Xe after the resident-budget and CPU-lane
  // tuning passes: the fused Gemma decode layer tail now wins on supported
  // q4_k_m and q6_k smoke shapes, so keep it on by default and leave an
  // opt-out for regressions on other machines via
  // LIZARD_NATIVE_GPU_FUSE_LAYER=0.
  static const bool env_enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_GPU_FUSE_LAYER");
    return value == nullptr || std::string(value) != "0";
  }();
  return env_enabled;
}

static bool gpu_gemma_attention_graph_enabled() {
  // Default ON since 2026-07-05: the resident attention/token-graph lane
  // passed the 200-token 5-run gate (median 4.82 tok/s vs the 3.31 target,
  // ~43.5 s total vs the 59.8 s target, TTFT ~2 s vs 6.92 s) with output
  // byte-identical to the trusted per-layer path after the attention score
  // fix. LIZARD_NATIVE_GPU_GRAPH=0 rolls back to the pre-graph lanes.
  static const bool env_enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_GPU_GRAPH");
    return value == nullptr || std::string(value) != "0";
  }();
  return env_enabled;
}

// Isolation gate for the complete-token graph: LIZARD_NATIVE_GPU_TOKEN_GRAPH=0
// keeps the per-layer resident attention path while skipping the whole-token
// submission, so parity failures can be pinned to one stage.
static bool gpu_gemma_token_graph_enabled() {
  static const bool env_enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_GPU_TOKEN_GRAPH");
    return value == nullptr || std::string(value) != "0";
  }();
  return env_enabled;
}

// Greedy decode fast path: when sampling is temperature 0 the GPU argmax is
// authoritative and the vocab-sized logits readback (about 1 MB per token)
// is skipped. LIZARD_NATIVE_GPU_GREEDY_FASTPATH=0 restores full readback.
static bool gpu_greedy_fastpath_enabled() {
  static const bool env_enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_GPU_GREEDY_FASTPATH");
    return value == nullptr || std::string(value) != "0";
  }();
  return env_enabled;
}

// Debug instrument: LIZARD_NATIVE_GPU_ATTN_VERIFY=1 runs the resident GPU
// layer AND the trusted CPU path for every layer, prints the per-layer
// max-abs difference to stderr, and continues with the CPU result so the
// comparison stays teacher-forced. Decode is slow in this mode; debug only.
static bool gpu_attn_verify_enabled() {
  static const bool env_enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_GPU_ATTN_VERIFY");
    return value != nullptr && std::string(value) == "1";
  }();
  return env_enabled;
}

// Unique identity per packed-input fill. Reusing buffer addresses as GPU
// input identities is unsafe because freed batches can be reallocated at the
// same address with different contents.
static const void *next_gpu_input_identity() {
  static std::atomic<std::uintptr_t> epoch {1};
  return reinterpret_cast<const void *>(epoch.fetch_add(1, std::memory_order_relaxed));
}

static thread_local bool g_decode_gpu_used = false;
static thread_local std::string g_decode_gpu_detail;
static thread_local std::uint64_t g_decode_gpu_use_count = 0;

static void reset_decode_gpu_trace() {
  g_decode_gpu_used = false;
  g_decode_gpu_detail.clear();
  g_decode_gpu_use_count = 0;
}

static void mark_decode_gpu_used(const std::string &detail) {
  g_decode_gpu_used = true;
  g_decode_gpu_use_count += 1;
  if (!detail.empty()) {
    g_decode_gpu_detail = detail;
  } else if (g_decode_gpu_detail.empty()) {
    g_decode_gpu_detail = "Direct3D 12 resident decode executed.";
  }
}

// Defined below, beside the CPU dot profile.
void record_decode_phase_since(const char *name, std::chrono::steady_clock::time_point started);
bool decode_phase_profile_enabled();

static bool decode_phase_metrics_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_PHASE_METRICS");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled || decode_phase_profile_enabled();
}

// For phases with several exit points, where a manual call before each return
// is easy to get wrong and silently under-reports.
struct DecodePhaseScope {
  const char *name;
  std::chrono::steady_clock::time_point started;
  explicit DecodePhaseScope(const char *phase_name)
    : name(phase_name), started(std::chrono::steady_clock::now()) {}
  ~DecodePhaseScope() { record_decode_phase_since(name, started); }
  DecodePhaseScope(const DecodePhaseScope &) = delete;
  DecodePhaseScope &operator=(const DecodePhaseScope &) = delete;
};

static void record_decode_phase_metric(
  SessionState &session,
  const std::string &label,
  std::int64_t layer_index,
  const std::chrono::steady_clock::time_point &started_at,
  std::uint64_t gpu_use_count_before,
  const std::string &gpu_detail_before) {
  // Detailed layer records contain labels/backend strings and can number in
  // the thousands for one response. Keep the always-on backend byte/time
  // counters cheap; materialize layer traces only for an explicit diagnostic.
  if (!decode_phase_metrics_enabled()) {
    return;
  }
  // Mirror every existing phase into the file profile. These metrics already
  // covered the Llama decode path at exactly the boundaries worth measuring -
  // QKV, attention CPU, attention output, FFN, fused layer tail - but they go
  // out as `decodePhaseMetrics` in the worker's generate response, which nothing
  // on the JS side reads, so in practice the data existed and was unreachable.
  // One call here instruments all thirteen sites.
  //
  // Microsecond resolution deliberately: the session metric floors elapsed_ms at
  // 1 ms, which would round most per-layer phases to the floor and make a
  // sub-millisecond phase indistinguishable from a 1 ms one.
  const auto finished_at = std::chrono::steady_clock::now();
  const std::uint64_t elapsed_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count()));
  const bool gpu_used = g_decode_gpu_use_count > gpu_use_count_before;
  std::string backend = gpu_used ? d3d12_batch_matmul_capabilities().backend_name : "cpu";
  if (!gpu_used && !gpu_detail_before.empty() && g_decode_gpu_detail != gpu_detail_before) {
    backend = d3d12_batch_matmul_capabilities().backend_name;
  }
  // Keyed by label AND backend, because "QKV is slow" and "QKV silently fell
  // back to the CPU lane" produce the same total and demand opposite fixes.
  // decode_qkv_fuse_blocked latches for the whole session on one refusal, so a
  // single early failure can move every later token's K/V projection to CPU
  // with nothing in the timings to say so.
  record_decode_phase_since((label + " [" + backend + "]").c_str(), started_at);
  session.decode_phase_metrics.push_back(SessionState::DecodePhaseMetric {
    label,
    layer_index,
    backend,
    elapsed_ms,
  });
}

// Defined below; the hybrid split calls it for the CPU half of a matvec.
static std::vector<float> matrix_vector_mul_tensor_rows(
  const LoadedModelState &state,
  const TensorInfo &tensor,
  std::uint64_t row_offset,
  std::uint64_t row_count,
  std::uint64_t row_width,
  const std::vector<float> &input,
  const std::string &label);

// Percent of a matvec's rows handed to CPU worker threads while the GPU runs
// the rest. 0 disables the split (the default until it is measured to win on a
// given machine): the right share depends on the ratio of the two engines'
// achieved bandwidth, which differs per adapter and per memory configuration.
static std::uint32_t hybrid_cpu_row_share(std::uint64_t row_count) {
  static const std::uint32_t percent = [] {
    char value[8] = {};
    if (GetEnvironmentVariableA("LIZARD_NATIVE_HYBRID_CPU_PERCENT", value, sizeof(value)) == 0) {
      return 0u;
    }
    const int parsed = std::atoi(value);
    return parsed > 0 && parsed < 100 ? static_cast<std::uint32_t>(parsed) : 0u;
  }();
  if (percent == 0) return 0;
  // Keep enough rows on each side to be worth the thread hop.
  const std::uint64_t rows = (row_count * percent) / 100u;
  return rows >= 64 && row_count - rows >= 64 ? static_cast<std::uint32_t>(rows) : 0u;
}

// See the call site in matrix_vector_mul_tensor for why this exists.
bool cpu_dot_profile_enabled() {
  static const bool on = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_CPU_PROFILE", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return on;
}

void record_cpu_dot_sample(std::uint64_t nanoseconds, std::uint64_t weight_bytes) {
  static std::atomic<std::uint64_t> total_ns {0};
  static std::atomic<std::uint64_t> total_bytes {0};
  static std::atomic<std::uint64_t> calls {0};
  total_ns.fetch_add(nanoseconds);
  total_bytes.fetch_add(weight_bytes);
  const std::uint64_t seen = calls.fetch_add(1) + 1;
  // Reported periodically rather than at exit: the runner is terminated by
  // signal when a server stops, so a static destructor never runs and the
  // numbers would be lost exactly when they are wanted.
  // Every 20, not every 200. At 200 a configuration that only routes a few
  // dozen matvecs through this lane writes NOTHING, which is indistinguishable
  // from "the lane is never used" - and that ambiguity cost a full measurement
  // cycle. An append every 20 matvecs is far cheaper than the matvecs it
  // reports on.
  // Also on the very first sample, so the file's mere existence proves the lane
  // executed. Without that, "no file" conflated "never called" with "called
  // fewer times than the reporting interval" - two conclusions with opposite
  // implications for where to optimise.
  if (seen != 1 && seen % 20 != 0) return;
  const double seconds = static_cast<double>(total_ns.load()) / 1e9;
  const double gib = static_cast<double>(total_bytes.load()) / (1024.0 * 1024.0 * 1024.0);
  // A file, not stderr. process-client.js captures the worker's stderr into a
  // bounded ring buffer that is only surfaced on failure, so anything the
  // runner prints there is discarded - which means "the profile printed
  // nothing" was never evidence of anything. A file cannot be swallowed.
  const char *const home = std::getenv("USERPROFILE");
  if (home == nullptr) {
    return;
  }
  const std::string path = std::string(home) + "\\.lizard-llm\\cpu-profile.log";
  std::FILE *const out = std::fopen(path.c_str(), "a");
  if (out == nullptr) {
    return;
  }
  std::fprintf(
    out,
    "[cpu-profile] row loop: %.2f s over %.3f GiB = %.2f GB/s across %llu matvecs\n",
    seconds,
    gib,
    seconds > 0.0 ? gib / seconds : 0.0,
    static_cast<unsigned long long>(seen));
  std::fclose(out);
}

// Named phase timing for the 49% of a decoded token that is NOT inside the
// fused layer tail: QKV projection, RoPE, the attention core, the LM head and
// sampling. The GPU stage stamps cover the tail and the tail only, and
// 08_WHERE_THE_TOKEN_ACTUALLY_GOES.md is explicit that the remaining work must
// be measured before any of it is changed - every guess this campaign made
// about where time goes has been wrong.
//
// Same three properties as the CPU dot profile, for the same reasons: a file
// sink (process-client.js swallows worker stderr), an absolute path, and a
// report on the FIRST record so the file existing proves the code ran.
bool decode_phase_profile_enabled() {
  static const bool on = [] {
    char value[8] = {};
    return GetEnvironmentVariableA("LIZARD_NATIVE_DECODE_PROFILE", value, sizeof(value)) != 0
      && value[0] == '1';
  }();
  return on;
}

void record_decode_phase_since(
  const char *name,
  std::chrono::steady_clock::time_point started) {
  if (!decode_phase_profile_enabled()) {
    return;
  }
  const double milliseconds = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  static std::mutex mutex;
  static std::map<std::string, std::pair<double, std::uint64_t>> totals;
  static std::uint64_t records = 0;
  std::lock_guard<std::mutex> guard(mutex);
  auto &entry = totals[name];
  entry.first += milliseconds;
  entry.second += 1;
  records += 1;
  if (records != 1 && records % 500 != 0) {
    return;
  }
  const char *const home = std::getenv("USERPROFILE");
  if (home == nullptr) {
    return;
  }
  const std::string path = std::string(home) + "\\.lizard-llm\\decode-profile.log";
  std::FILE *const out = std::fopen(path.c_str(), "a");
  if (out == nullptr) {
    return;
  }
  double total = 0.0;
  for (const auto &phase : totals) {
    total += phase.second.first;
  }
  std::fprintf(out, "[decode-profile] records=%llu\n",
    static_cast<unsigned long long>(records));
  for (const auto &phase : totals) {
    std::fprintf(out,
      "  %-22s %10.1f ms total  %8llu calls  %7.3f ms/call  %5.1f%%\n",
      phase.first.c_str(),
      phase.second.first,
      static_cast<unsigned long long>(phase.second.second),
      phase.second.second > 0
        ? phase.second.first / static_cast<double>(phase.second.second)
        : 0.0,
      total > 0.0 ? phase.second.first * 100.0 / total : 0.0);
  }
  std::fclose(out);
}

static std::vector<float> matrix_vector_mul_tensor(
  const LoadedModelState &state,
  const TensorInfo &tensor,
  const MatrixShape &shape,
  const std::vector<float> &input,
  const std::string &label) {
  if (shape.row_width != input.size()) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " width does not match the input vector size.");
  }
  if (shape.row_width == 0 || shape.row_count == 0 || shape.row_count > (std::numeric_limits<std::uint64_t>::max() / shape.row_width)) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " has an invalid tensor shape.");
  }
  if (shape.row_width * shape.row_count != tensor.element_count) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " tensor payload does not match the advertised tensor shape.");
  }

  // Required GPU mode routes every supported decode projection through D3D12
  // and fails explicitly instead of entering the extremely slow CPU lane.
  if (gpu_decode_lane_enabled_for_model(state)
    && (g_gpu_decode_required || shape.row_count * shape.row_width >= 4194304)) {
    D3d12WeightFormat gpu_format = D3d12WeightFormat::F32;
    if (d3d12_weight_format_supported(tensor.ggml_type, shape.row_width, &gpu_format)) {
      const std::uint8_t *weight_data = tensor_data_ptr(state, tensor);
      D3d12ResidentWeightRef op;
      op.weight_key = weight_data;
      op.format = gpu_format;
      op.weight_data = weight_data;
      op.weight_bytes = static_cast<std::size_t>(tensor.byte_length);
      op.row_count = static_cast<std::uint32_t>(shape.row_count);
      op.row_width = static_cast<std::uint32_t>(shape.row_width);

      // Hybrid split (opt-in). Decode is memory-bandwidth-bound on both engines:
      // measured 12.9 GB/s on this iGPU and 16.3 GB/s from llama.cpp's CPU
      // kernels on the same DDR4. Neither alone reaches the 17.7 GB/s a 1.10x
      // result needs, so give the tail rows to CPU worker threads and let them
      // run while the GPU works on the rest. The GPU entry point blocks its
      // calling thread, so the overlap comes from the CPU rows executing on
      // OTHER threads - no async D3D12 submission is required.
      //
      // Safe against the resident cache: entries are keyed on the weight
      // pointer and store only the buffer and its byte count, so staging the
      // whole tensor and dispatching a row prefix is well defined. Rows are
      // contiguous, so the CPU takes the suffix with a plain row offset.
      const std::uint32_t cpu_rows = hybrid_cpu_row_share(shape.row_count);
      if (cpu_rows > 0 && cpu_rows < shape.row_count) {
        const std::uint64_t gpu_rows = shape.row_count - cpu_rows;
        op.row_count = static_cast<std::uint32_t>(gpu_rows);
        std::vector<float> cpu_tail;
        std::exception_ptr cpu_error;
        std::thread cpu_worker([&]() {
          try {
            cpu_tail = matrix_vector_mul_tensor_rows(
              state, tensor, gpu_rows, cpu_rows, shape.row_width, input, label);
          } catch (...) {
            cpu_error = std::current_exception();
          }
        });
        std::vector<float> gpu_head;
        std::string split_error;
        const bool gpu_ok = d3d12_batch_matmul_resident(
          &op, 1, input, 1, next_gpu_input_identity(), &gpu_head, &split_error);
        cpu_worker.join();
        if (cpu_error) std::rethrow_exception(cpu_error);
        if (gpu_ok && gpu_head.size() == gpu_rows && cpu_tail.size() == cpu_rows) {
          mark_decode_gpu_used(
            "Direct3D 12 hybrid decode matvec (GPU head + CPU tail) on "
            + d3d12_batch_matmul_capabilities().adapter_name + ".");
          gpu_head.insert(gpu_head.end(), cpu_tail.begin(), cpu_tail.end());
          return gpu_head;
        }
        // Any refusal falls through to the whole-tensor GPU path below.
        op.row_count = static_cast<std::uint32_t>(shape.row_count);
      }

      std::vector<float> flat_output;
      std::string gpu_error;
      if (d3d12_batch_matmul_resident(&op, 1, input, 1, next_gpu_input_identity(), &flat_output, &gpu_error)) {
        mark_decode_gpu_used(
          "Direct3D 12 resident decode matvec executed on "
          + d3d12_batch_matmul_capabilities().adapter_name + ".");
        return flat_output;
      }
      if (g_gpu_decode_required) {
        throw_required_gpu_decode_failure(label, gpu_error);
      }
    } else if (g_gpu_decode_required) {
      throw_required_gpu_decode_failure(label, "the tensor format or row width is unsupported");
    }
  }

  // Enhancement Run 06 base lane. The pinned ggml backend owns the complete
  // CPU matvec when selected: an error is explicit and never falls through to
  // native code under a misleading backend label. Lizard still owns model
  // mapping, graph/session orchestration, KV state, batching, and RPC.
  const std::string ggml_selection_error = ggml_cpu_backend_selection_error();
  if (!ggml_selection_error.empty()) {
    throw WorkerError("NATIVE_GGML_BACKEND_ERROR", ggml_selection_error);
  }
  if (ggml_cpu_backend_selected()) {
    std::vector<float> output;
    std::string ggml_error;
    const bool profiling = cpu_dot_profile_enabled();
    const auto profile_started = profiling
      ? std::chrono::steady_clock::now()
      : std::chrono::steady_clock::time_point{};
    if (!ggml_cpu_backend_matvec(
        tensor.ggml_type,
        shape.row_count,
        shape.row_width,
        tensor_data_ptr(state, tensor),
        static_cast<std::size_t>(tensor.byte_length),
        input.data(),
        input.size(),
        max_cpu_row_workers(),
        &output,
        &ggml_error)) {
      throw WorkerError(
        "NATIVE_GGML_BACKEND_ERROR",
        label + " failed in pinned ggml CPU backend: " + ggml_error,
        JsonObject {
          {"ggmlCommit", JsonValue(k_lizard_ggml_commit)},
          {"tensor", JsonValue(tensor.name)},
          {"ggmlType", JsonValue(static_cast<std::uint64_t>(tensor.ggml_type))},
        });
    }
    if (profiling) {
      record_cpu_dot_sample(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - profile_started).count(),
        static_cast<std::uint64_t>(tensor.byte_length));
    }
    return output;
  }

#ifdef LIZARD_NATIVE_AVX2
  if (tensor.ggml_type == 14
    && shape.row_width % k_qk_k == 0
    && q6_k_q8_k_fastpath_enabled()) {
    const std::vector<Q8KActivationBlock> activations = quantize_q8_k_activation(input);
    if (!activations.empty()) {
      std::vector<float> output(static_cast<std::size_t>(shape.row_count), 0.0f);
      const std::uint64_t total_work = shape.row_count * shape.row_width;
      const std::uint64_t worker_count = total_work >= 262144
        ? std::min<std::uint64_t>(
          static_cast<std::uint64_t>(max_cpu_row_workers()),
          std::max<std::uint64_t>(1, shape.row_count / 64))
        : 1;
      const std::uint8_t *weight_data = tensor_data_ptr(state, tensor);
      const std::size_t row_bytes = activations.size() * 210u;
      const auto compute_rows = [&](std::uint64_t start_row, std::uint64_t end_row) {
        for (std::uint64_t row = start_row; row < end_row; row += 1) {
          output[static_cast<std::size_t>(row)] = q6_k_q8_k_row_dot_avx2(
            weight_data + static_cast<std::size_t>(row) * row_bytes,
            activations);
        }
      };
      const bool profiling = cpu_dot_profile_enabled();
      const auto profile_started = profiling
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
      if (worker_count <= 1) {
        compute_rows(0, shape.row_count);
      } else {
        persistent_row_workers().execute(shape.row_count, worker_count, compute_rows);
      }
      if (profiling) {
        record_cpu_dot_sample(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - profile_started).count(),
          static_cast<std::uint64_t>(shape.row_count * row_bytes));
      }
      return output;
    }
  }
  if (tensor.ggml_type == 12
    && shape.row_width % k_qk_k == 0
    && q4_k_q8_k_fastpath_enabled()) {
    const std::vector<Q8KActivationBlock> activations = quantize_q8_k_activation(input);
    if (!activations.empty()) {
      std::vector<float> output(static_cast<std::size_t>(shape.row_count), 0.0f);
      const std::uint64_t total_work = shape.row_count * shape.row_width;
      const std::uint64_t worker_count = total_work >= 262144
        ? std::min<std::uint64_t>(
          static_cast<std::uint64_t>(max_cpu_row_workers()),
          std::max<std::uint64_t>(1, shape.row_count / 64))
        : 1;
      const std::uint8_t *weight_data = tensor_data_ptr(state, tensor);
      const std::size_t row_bytes = activations.size() * 144u;
      const auto compute_rows = [&](std::uint64_t start_row, std::uint64_t end_row) {
        for (std::uint64_t row = start_row; row < end_row; row += 1) {
          output[static_cast<std::size_t>(row)] = q4_k_q8_k_row_dot_avx2(
            weight_data + static_cast<std::size_t>(row) * row_bytes,
            activations);
        }
      };
      const bool profiling = cpu_dot_profile_enabled();
      const auto profile_started = profiling
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
      if (worker_count <= 1) {
        compute_rows(0, shape.row_count);
      } else {
        persistent_row_workers().execute(shape.row_count, worker_count, compute_rows);
      }
      if (profiling) {
        record_cpu_dot_sample(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - profile_started).count(),
          static_cast<std::uint64_t>(shape.row_count * row_bytes));
      }
      return output;
    }
  }
#endif

  std::vector<float> output(static_cast<std::size_t>(shape.row_count), 0.0f);
  const std::uint64_t total_work = shape.row_count * shape.row_width;
  const std::uint64_t worker_count = total_work >= 262144
    ? std::min<std::uint64_t>(
      static_cast<std::uint64_t>(max_cpu_row_workers()),
      std::max<std::uint64_t>(1, shape.row_count / 64))
    : 1;
  const auto compute_rows = [&](std::uint64_t start_row, std::uint64_t end_row) {
    for (std::uint64_t row = start_row; row < end_row; row += 1) {
      output[static_cast<std::size_t>(row)] = static_cast<float>(tensor_range_dot_f32(
        state,
        tensor,
        row * shape.row_width,
        shape.row_width,
        input,
        label));
    }
  };

  // CPU decode attribution, under LIZARD_NATIVE_CPU_PROFILE=1. Six informed
  // guesses on this lane produced small or null results, so measure: wall time
  // spent inside the row loop against the quantised weight bytes it swept.
  //
  // If the GB/s here approaches what llama.cpp gets from this bus, the rest of
  // the gap is OUTSIDE the kernel - dispatch, scheduling, allocation - and more
  // vectorising will not help. If it stays far below, the kernel is still the
  // problem and this says by how much. Timed once per matvec, not per row, so
  // the measurement cannot distort what it measures.
  const bool profiling = cpu_dot_profile_enabled();
  const auto profile_started = profiling
    ? std::chrono::steady_clock::now()
    : std::chrono::steady_clock::time_point{};

  if (worker_count <= 1) {
    compute_rows(0, shape.row_count);
  } else {
    persistent_row_workers().execute(shape.row_count, worker_count, compute_rows);
  }

  if (profiling) {
    // Q4_K packs 256 weights into 144 bytes; close enough across formats for an
    // order-of-magnitude read, and the label says which tensor it came from.
    record_cpu_dot_sample(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - profile_started).count(),
      (shape.row_count * shape.row_width * 144u) / 256u);
  }
  return output;
}

// Matrix-vector product over a row window of a (possibly 3D) tensor. MoE
// expert tensors pack experts contiguously (expert e = rows
// [e*rows_per_expert, (e+1)*rows_per_expert)), so a row offset plus the
// existing range-dot dequantization covers per-expert projections without a
// new dequant path.
static std::vector<float> matrix_vector_mul_tensor_rows(
  const LoadedModelState &state,
  const TensorInfo &tensor,
  std::uint64_t row_offset,
  std::uint64_t row_count,
  std::uint64_t row_width,
  const std::vector<float> &input,
  const std::string &label) {
  if (row_width != input.size() || row_width == 0 || row_count == 0) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " width does not match the input vector size.");
  }
  if ((row_offset + row_count) * row_width > tensor.element_count) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " row window extends beyond the tensor payload.");
  }
  std::vector<float> output(static_cast<std::size_t>(row_count), 0.0f);
  const auto compute_rows = [&](std::uint64_t start_row, std::uint64_t end_row) {
    for (std::uint64_t row = start_row; row < end_row; row += 1) {
      output[static_cast<std::size_t>(row)] = static_cast<float>(tensor_range_dot_f32(
        state,
        tensor,
        (row_offset + row) * row_width,
        row_width,
        input,
        label));
    }
  };
  const std::uint64_t total_work = row_count * row_width;
  const std::uint64_t worker_count = total_work >= 262144
    ? std::min<std::uint64_t>(
      static_cast<std::uint64_t>(max_cpu_row_workers()),
      std::max<std::uint64_t>(1, row_count / 64))
    : 1;
  // Same instrument as matrix_vector_mul_tensor, and it has to be here too:
  // this is the function the hybrid CPU+GPU split calls for the CPU share, so
  // with a working GPU lane the whole-tensor entry point above is never reached
  // and the profile stayed empty - which read as "the CPU kernel is not being
  // used" when in fact it was simply unmeasured.
  const bool profiling = cpu_dot_profile_enabled();
  const auto profile_started = profiling
    ? std::chrono::steady_clock::now()
    : std::chrono::steady_clock::time_point{};

  if (worker_count <= 1) {
    compute_rows(0, row_count);
  } else {
    persistent_row_workers().execute(row_count, worker_count, compute_rows);
  }

  if (profiling) {
    record_cpu_dot_sample(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - profile_started).count(),
      (row_count * row_width * 144u) / 256u);
  }
  return output;
}

static bool build_resident_ref(
  const LoadedModelState &state,
  const TensorInfo &tensor,
  const MatrixShape &shape,
  D3d12ResidentWeightRef &ref) {
  D3d12WeightFormat format = D3d12WeightFormat::F32;
  if (shape.row_width == 0
    || shape.row_count == 0
    || shape.row_width * shape.row_count != tensor.element_count
    || !d3d12_weight_format_supported(tensor.ggml_type, shape.row_width, &format)) {
    return false;
  }
  const std::uint8_t *weight_data = tensor_data_ptr(state, tensor);
  ref.weight_key = weight_data;
  ref.format = format;
  ref.weight_data = weight_data;
  ref.weight_bytes = static_cast<std::size_t>(tensor.byte_length);
  ref.row_count = static_cast<std::uint32_t>(shape.row_count);
  ref.row_width = static_cast<std::uint32_t>(shape.row_width);
  return true;
}

using FloatBatch = std::vector<std::vector<float>>;

// Whole decode FFN block (gate/up matmuls, gelu-mul, down matmul) in one GPU
// submission with device-only intermediates.
// Whole Llama decode layer tail in ONE submission: attention-output matvec,
// residual add, FFN RMS-norm, gate/up matvecs, SwiGLU, down matvec, residual
// add. Replaces two submissions (attention output, FFN chain), two CPU vector
// passes and two device readbacks with one submission and one readback.
//
// Measured before this existed: 85 GPU submissions per decoded token on a
// 28-layer model, each an ExecuteCommandLists plus a fence wait, which on an
// integrated adapter is where essentially all the decode time went.
static bool try_llama_layer_tail_resident(
  const LoadedModelState &state,
  const TensorInfo &attn_out_tensor,
  const MatrixShape &attn_out_shape,
  const TensorInfo &gate_tensor,
  const MatrixShape &gate_shape,
  const TensorInfo &up_tensor,
  const MatrixShape &up_shape,
  const TensorInfo &down_tensor,
  const MatrixShape &down_shape,
  const std::vector<float> &attention_context,
  const std::vector<float> &hidden_state,
  const std::vector<float> &ffn_norm_weights,
  float rms_epsilon,
  std::vector<float> &hidden_out) {
  if (!gpu_decode_lane_enabled_for_model(state)
    || !gpu_fuse_ffn_enabled()
    || attn_out_shape.row_width != attention_context.size()
    || attn_out_shape.row_count != hidden_state.size()
    || ffn_norm_weights.size() != hidden_state.size()) {
    return false;
  }
  D3d12ResidentWeightRef attn_out_ref;
  D3d12ResidentWeightRef gate_ref;
  D3d12ResidentWeightRef up_ref;
  D3d12ResidentWeightRef down_ref;
  if (!build_resident_ref(state, attn_out_tensor, attn_out_shape, attn_out_ref)
    || !build_resident_ref(state, gate_tensor, gate_shape, gate_ref)
    || !build_resident_ref(state, up_tensor, up_shape, up_ref)
    || !build_resident_ref(state, down_tensor, down_shape, down_ref)) {
    if (g_gpu_decode_required) {
      throw_required_gpu_decode_failure(
        "Fused Llama layer tail", "one or more resident weight formats are unsupported");
    }
    return false;
  }
  std::string gpu_error;
  if (!d3d12_llama_layer_tail_resident(
      attn_out_ref,
      gate_ref,
      up_ref,
      down_ref,
      attention_context,
      hidden_state,
      ffn_norm_weights,
      rms_epsilon,
      &hidden_out,
      &gpu_error)) {
    if (g_gpu_decode_required) {
      throw_required_gpu_decode_failure("Fused Llama layer tail", gpu_error);
    }
    return false;
  }
  // Same guard the FFN chain carries: a non-finite result means the fused path
  // disagreed with the model state, so refuse it and let the split path answer
  // rather than propagate a poisoned hidden state through the tower.
  for (const float value : hidden_out) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  mark_decode_gpu_used(
    "Direct3D 12 resident decode Llama layer tail executed on "
    + d3d12_batch_matmul_capabilities().adapter_name + ".");
  return true;
}

static bool try_llama_layer_tail_batch_resident(
  const LoadedModelState &state,
  const TensorInfo &attn_out_tensor,
  const MatrixShape &attn_out_shape,
  const TensorInfo &gate_tensor,
  const MatrixShape &gate_shape,
  const TensorInfo &up_tensor,
  const MatrixShape &up_shape,
  const TensorInfo &down_tensor,
  const MatrixShape &down_shape,
  const FloatBatch &attention_contexts,
  const FloatBatch &hidden_states,
  const std::vector<float> &ffn_norm_weights,
  float rms_epsilon,
  FloatBatch &hidden_out) {
  if (!gpu_decode_lane_enabled_for_model(state)
    || !gpu_fuse_ffn_enabled()
    || !gpu_fuse_layer_enabled()
    || attention_contexts.size() < 2
    || attention_contexts.size() != hidden_states.size()
    || attention_contexts.size() > std::numeric_limits<std::uint32_t>::max()
    || ffn_norm_weights.size() != attn_out_shape.row_count) {
    return false;
  }
  for (std::size_t row = 0; row < attention_contexts.size(); row += 1) {
    if (attention_contexts[row].size() != attn_out_shape.row_width
      || hidden_states[row].size() != attn_out_shape.row_count) {
      return false;
    }
  }
  D3d12ResidentWeightRef attn_out_ref;
  D3d12ResidentWeightRef gate_ref;
  D3d12ResidentWeightRef up_ref;
  D3d12ResidentWeightRef down_ref;
  if (!build_resident_ref(state, attn_out_tensor, attn_out_shape, attn_out_ref)
    || !build_resident_ref(state, gate_tensor, gate_shape, gate_ref)
    || !build_resident_ref(state, up_tensor, up_shape, up_ref)
    || !build_resident_ref(state, down_tensor, down_shape, down_ref)) {
    return false;
  }

  std::vector<float> packed_contexts;
  std::vector<float> packed_hidden;
  packed_contexts.reserve(attention_contexts.size() * attention_contexts.front().size());
  packed_hidden.reserve(hidden_states.size() * hidden_states.front().size());
  for (const std::vector<float> &row : attention_contexts) {
    packed_contexts.insert(packed_contexts.end(), row.begin(), row.end());
  }
  for (const std::vector<float> &row : hidden_states) {
    packed_hidden.insert(packed_hidden.end(), row.begin(), row.end());
  }

  std::vector<float> packed_output;
  std::string gpu_error;
  if (!d3d12_llama_layer_tail_batch_resident(
      attn_out_ref,
      gate_ref,
      up_ref,
      down_ref,
      packed_contexts,
      packed_hidden,
      ffn_norm_weights,
      static_cast<std::uint32_t>(hidden_states.size()),
      rms_epsilon,
      &packed_output,
      &gpu_error)) {
    if (g_gpu_decode_required) {
      throw_required_gpu_decode_failure("Fused batched Llama layer tail", gpu_error);
    }
    return false;
  }
  const std::size_t hidden_width = static_cast<std::size_t>(attn_out_shape.row_count);
  if (packed_output.size() != hidden_states.size() * hidden_width) {
    return false;
  }
  hidden_out.assign(hidden_states.size(), std::vector<float>(hidden_width, 0.0f));
  for (std::size_t row = 0; row < hidden_states.size(); row += 1) {
    std::memcpy(
      hidden_out[row].data(),
      packed_output.data() + row * hidden_width,
      hidden_width * sizeof(float));
    for (const float value : hidden_out[row]) {
      if (!std::isfinite(value)) {
        return false;
      }
    }
  }
  mark_decode_gpu_used(
    "Direct3D 12 resident batched Llama layer tail executed on "
    + d3d12_batch_matmul_capabilities().adapter_name + ".");
  return true;
}

static bool try_ffn_chain_matvec_resident(
  const LoadedModelState &state,
  const TensorInfo &gate_tensor,
  const MatrixShape &gate_shape,
  const TensorInfo &up_tensor,
  const MatrixShape &up_shape,
  const TensorInfo &down_tensor,
  const MatrixShape &down_shape,
  const std::vector<float> &input,
  std::vector<float> &output,
  bool use_silu = false) {
  if (!gpu_decode_lane_enabled_for_model(state) || !gpu_fuse_ffn_enabled() || gate_shape.row_width != input.size()) {
    return false;
  }
  D3d12ResidentWeightRef gate_ref;
  D3d12ResidentWeightRef up_ref;
  D3d12ResidentWeightRef down_ref;
  if (!build_resident_ref(state, gate_tensor, gate_shape, gate_ref)
    || !build_resident_ref(state, up_tensor, up_shape, up_ref)
    || !build_resident_ref(state, down_tensor, down_shape, down_ref)) {
    if (g_gpu_decode_required) {
      throw_required_gpu_decode_failure("Fused FFN", "one or more resident weight formats are unsupported");
    }
    return false;
  }
  std::string gpu_error;
  const bool chain_ok = use_silu
    ? d3d12_ffn_silu_glu_resident(
      gate_ref, up_ref, down_ref, input, 1, next_gpu_input_identity(), &output, &gpu_error)
    : d3d12_ffn_gelu_glu_resident(
      gate_ref, up_ref, down_ref, input, 1, next_gpu_input_identity(), &output, &gpu_error);
  if (!chain_ok) {
    if (g_gpu_decode_required) {
      throw_required_gpu_decode_failure("Fused FFN", gpu_error);
    }
    return false;
  }
  mark_decode_gpu_used(
    "Direct3D 12 resident decode FFN chain executed on "
    + d3d12_batch_matmul_capabilities().adapter_name + ".");
  for (std::size_t index = 0; index < output.size(); index += 1) {
    if (!std::isfinite(output[index])) {
      std::fprintf(
        stderr,
        "lizard-native: fused decode FFN chain produced non-finite value at row %zu; "
        "falling back to unfused projections.\n",
        index);
      if (g_gpu_decode_required) {
        throw_required_gpu_decode_failure("Fused FFN", "the GPU result contained a non-finite value");
      }
      return false;
    }
  }
  return true;
}

// Runs up to four single-vector projections that share one input in a single
// GPU submission. Each ExecuteCommandLists+fence round-trip costs far more
// than the matvecs themselves during decode, so fusing q/k/v cuts the
// dominant per-token overhead.
static bool try_multi_matvec_resident(
  const LoadedModelState &state,
  const TensorInfo *const tensors[],
  const MatrixShape *const shapes[],
  std::size_t op_count,
  const std::vector<float> &input,
  std::vector<float> *const outputs[]) {
  if (!gpu_decode_lane_enabled_for_model(state) || !gpu_fuse_qkv_enabled() || op_count == 0 || op_count > 4) {
    return false;
  }
  D3d12ResidentWeightRef ops[4];
  std::uint64_t total_elements = 0;
  for (std::size_t op_index = 0; op_index < op_count; op_index += 1) {
    const TensorInfo &tensor = *tensors[op_index];
    const MatrixShape &shape = *shapes[op_index];
    if (shape.row_width != input.size()
      || shape.row_width == 0
      || shape.row_count == 0
      || shape.row_width * shape.row_count != tensor.element_count) {
      if (g_gpu_decode_required) {
        throw_required_gpu_decode_failure("Fused QKV", "a tensor shape does not match the shared input");
      }
      return false;
    }
    D3d12WeightFormat format = D3d12WeightFormat::F32;
    if (!d3d12_weight_format_supported(tensor.ggml_type, shape.row_width, &format)) {
      if (g_gpu_decode_required) {
        throw_required_gpu_decode_failure("Fused QKV", "one or more tensor formats are unsupported");
      }
      return false;
    }
    const std::uint8_t *weight_data = tensor_data_ptr(state, tensor);
    ops[op_index].weight_key = weight_data;
    ops[op_index].format = format;
    ops[op_index].weight_data = weight_data;
    ops[op_index].weight_bytes = static_cast<std::size_t>(tensor.byte_length);
    ops[op_index].row_count = static_cast<std::uint32_t>(shape.row_count);
    ops[op_index].row_width = static_cast<std::uint32_t>(shape.row_width);
    total_elements += shape.row_count * shape.row_width;
  }
  if (total_elements < 4194304) {
    if (g_gpu_decode_required) {
      throw_required_gpu_decode_failure("Fused QKV", "the fused projection is below the current GPU work threshold");
    }
    return false;
  }
  std::vector<float> flat_outputs[4];
  std::string gpu_error;
  if (!d3d12_batch_matmul_resident(
    ops,
    op_count,
    input,
    1,
    next_gpu_input_identity(),
    flat_outputs,
    &gpu_error)) {
    if (g_gpu_decode_required) {
      throw_required_gpu_decode_failure("Fused QKV", gpu_error);
    }
    return false;
  }
  mark_decode_gpu_used(
    "Direct3D 12 resident fused decode QKV executed on "
    + d3d12_batch_matmul_capabilities().adapter_name + ".");
  for (std::size_t op_index = 0; op_index < op_count; op_index += 1) {
    *outputs[op_index] = std::move(flat_outputs[op_index]);
  }
  return true;
}

// Phase 0c experiment: several independent CPU projections share one cached
// ggml graph and one activation tensor. This removes graph-plan dispatch and
// thread-pool synchronization between Q/K/V and gate/up without changing
// Lizard's model, KV, attention, sampling, or lifecycle ownership.
static bool try_multi_matvec_pinned_ggml(
  const LoadedModelState &state,
  const TensorInfo *const tensors[],
  const MatrixShape *const shapes[],
  std::size_t op_count,
  const std::vector<float> &input,
  std::vector<float> *const outputs[],
  const std::string &label) {
  if (!ggml_cpu_backend_selected() || !ggml_cpu_backend_grouped_graphs_enabled()) {
    return false;
  }
  const std::string selection_error = ggml_cpu_backend_selection_error();
  if (!selection_error.empty()) {
    throw WorkerError("NATIVE_GGML_BACKEND_ERROR", selection_error);
  }
  if (op_count < 2 || op_count > 4) {
    throw WorkerError("NATIVE_GGML_BACKEND_ERROR", label + " requires two through four projections.");
  }
  GgmlCpuMatvecOperation operations[4];
  std::uint64_t total_weight_bytes = 0;
  for (std::size_t index = 0; index < op_count; index += 1) {
    const TensorInfo &tensor = *tensors[index];
    const MatrixShape &shape = *shapes[index];
    if (shape.row_width != input.size()
      || shape.row_width == 0
      || shape.row_count == 0
      || shape.row_count > std::numeric_limits<std::uint64_t>::max() / shape.row_width
      || shape.row_width * shape.row_count != tensor.element_count) {
      throw WorkerError("NATIVE_GGML_BACKEND_ERROR", label + " tensor shape does not match the shared input.");
    }
    operations[index] = GgmlCpuMatvecOperation {
      tensor.ggml_type,
      shape.row_count,
      shape.row_width,
      tensor_data_ptr(state, tensor),
      static_cast<std::size_t>(tensor.byte_length),
      outputs[index],
    };
    total_weight_bytes += tensor.byte_length;
  }
  const bool profiling = cpu_dot_profile_enabled();
  const auto profile_started = profiling
    ? std::chrono::steady_clock::now()
    : std::chrono::steady_clock::time_point {};
  std::string error;
  if (!ggml_cpu_backend_multi_matvec(
      operations,
      op_count,
      input.data(),
      input.size(),
      max_cpu_row_workers(),
      &error)) {
    throw WorkerError(
      "NATIVE_GGML_BACKEND_ERROR",
      label + " failed in pinned ggml grouped graph: " + error,
      JsonObject {{"ggmlCommit", JsonValue(k_lizard_ggml_commit)}});
  }
  if (profiling) {
    record_cpu_dot_sample(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - profile_started).count(),
      total_weight_bytes);
  }
  return true;
}

static bool try_ffn_swiglu_pinned_ggml(
  const LoadedModelState &state,
  const TensorInfo &gate_tensor,
  const MatrixShape &gate_shape,
  const TensorInfo &up_tensor,
  const MatrixShape &up_shape,
  const TensorInfo &down_tensor,
  const MatrixShape &down_shape,
  const std::vector<float> &input,
  std::vector<float> &output) {
  if (!ggml_cpu_backend_selected() || !ggml_cpu_backend_fused_ffn_enabled()) {
    return false;
  }
  const std::string selection_error = ggml_cpu_backend_selection_error();
  if (!selection_error.empty()) {
    throw WorkerError("NATIVE_GGML_BACKEND_ERROR", selection_error);
  }
  const GgmlCpuWeight gate {
    gate_tensor.ggml_type,
    gate_shape.row_count,
    gate_shape.row_width,
    tensor_data_ptr(state, gate_tensor),
    static_cast<std::size_t>(gate_tensor.byte_length),
  };
  const GgmlCpuWeight up {
    up_tensor.ggml_type,
    up_shape.row_count,
    up_shape.row_width,
    tensor_data_ptr(state, up_tensor),
    static_cast<std::size_t>(up_tensor.byte_length),
  };
  const GgmlCpuWeight down {
    down_tensor.ggml_type,
    down_shape.row_count,
    down_shape.row_width,
    tensor_data_ptr(state, down_tensor),
    static_cast<std::size_t>(down_tensor.byte_length),
  };
  const bool profiling = cpu_dot_profile_enabled();
  const auto profile_started = profiling
    ? std::chrono::steady_clock::now()
    : std::chrono::steady_clock::time_point {};
  std::string error;
  if (!ggml_cpu_backend_ffn_swiglu(
      gate,
      up,
      down,
      input.data(),
      input.size(),
      max_cpu_row_workers(),
      &output,
      &error)) {
    throw WorkerError(
      "NATIVE_GGML_BACKEND_ERROR",
      "Llama fused FFN failed in pinned ggml: " + error,
      JsonObject {{"ggmlCommit", JsonValue(k_lizard_ggml_commit)}});
  }
  if (profiling) {
    record_cpu_dot_sample(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - profile_started).count(),
      gate_tensor.byte_length + up_tensor.byte_length + down_tensor.byte_length);
  }
  return true;
}

static bool batch_matmul_gpu_eligible(
  const MatrixShape &shape,
  const FloatBatch &inputs,
  const BatchExecutionConfig *execution_config) {
  const std::uint64_t total_work = shape.row_count * shape.row_width * inputs.size();
  const bool large_projection = shape.row_count >= 4096 || shape.row_width >= 4096;
  const bool medium_projection = shape.row_count >= 1024 && total_work >= 134217728;
  return execution_config != nullptr
    && execution_config->allow_hybrid_gpu
    && inputs.size() > 1
    && shape.row_width >= 2048
    && (large_projection || medium_projection)
    && total_work >= 67108864;
}

static bool batch_matmul_gpu_resident_eligible(
  const TensorInfo &tensor,
  const MatrixShape &shape,
  const FloatBatch &inputs,
  const BatchExecutionConfig *execution_config,
  D3d12WeightFormat *format) {
  // Resident weights stay on the GPU in their quantized storage format, so a
  // projection pays no per-dispatch dequant or weight upload. That makes the
  // GPU lane profitable at much smaller shapes than the legacy tiled path.
  // Thresholds lowered 2026-07-06: the Gemma per-layer (PLE) gate
  // (256x2560, ~10M work at batch 15) and PLE projection (2560x256) sat
  // just under the old 16M/width-1024 gates and burned ~1.07 s of every
  // prefill on the CPU lane, even though the decode token graph already
  // runs these exact shapes on the GPU per token.
  return execution_config != nullptr
    && execution_config->allow_hybrid_gpu
    && inputs.size() > 1
    && shape.row_width >= 256
    && shape.row_count >= 256
    && shape.row_count * shape.row_width * inputs.size() >= 8388608
    && d3d12_weight_format_supported(tensor.ggml_type, shape.row_width, format);
}

static void record_prefill_projection_metric(
  BatchExecutionConfig *execution_config,
  const std::string &label,
  const MatrixShape &shape,
  std::uint64_t batch_count,
  const std::string &backend,
  std::uint64_t elapsed_ms) {
  if (execution_config == nullptr || execution_config->prefill_projection_metrics == nullptr) {
    return;
  }
  execution_config->prefill_projection_metrics->push_back(SessionState::PrefillProjectionMetric {
    label,
    execution_config->current_layer_index,
    batch_count,
    shape.row_count,
    shape.row_width,
    backend,
    elapsed_ms,
  });
}

static std::uint64_t choose_row_worker_count(
  std::uint64_t total_work,
  std::uint64_t row_count,
  std::uint64_t work_threshold = 262144,
  std::uint64_t min_rows_per_worker = 64) {
  if (total_work < work_threshold) {
    return 1;
  }
  return std::min<std::uint64_t>(
    static_cast<std::uint64_t>(max_cpu_row_workers()),
    std::max<std::uint64_t>(1, row_count / min_rows_per_worker));
}

static void fill_cached_gpu_packed_inputs(
  const MatrixShape &shape,
  const FloatBatch &inputs,
  BatchExecutionConfig &execution_config) {
  // Each fill gets a fresh identity: uploading the packed batch again is
  // cheap; stale inputs from a recycled buffer address are not.
  execution_config.cached_gpu_input_owner = next_gpu_input_identity();
  execution_config.cached_gpu_batch_count = static_cast<std::uint32_t>(inputs.size());
  execution_config.cached_gpu_input_width = static_cast<std::uint32_t>(shape.row_width);
  execution_config.cached_gpu_packed_inputs.assign(
    static_cast<std::size_t>(shape.row_width) * inputs.size(),
    0.0f);
  for (std::size_t batch_index = 0; batch_index < inputs.size(); batch_index += 1) {
    std::memcpy(
      execution_config.cached_gpu_packed_inputs.data() + batch_index * static_cast<std::size_t>(shape.row_width),
      inputs[batch_index].data(),
      static_cast<std::size_t>(shape.row_width) * sizeof(float));
  }
}

static void dequantize_tensor_row_tile_slice_into_f32(
  const LoadedModelState &state,
  const TensorInfo &tensor,
  const MatrixShape &shape,
  std::uint64_t start_row,
  std::uint64_t row_count,
  std::vector<float> &tile_weights,
  std::uint64_t destination_row_offset) {
  if (row_count == 0 || shape.row_width == 0 || tile_weights.empty()) {
    return;
  }

  const std::uint64_t destination_base = destination_row_offset * shape.row_width;
  const auto dequantize_rows = [&](std::uint64_t local_start_row, std::uint64_t local_end_row) {
    std::vector<float> row_weights;
    row_weights.reserve(static_cast<std::size_t>(shape.row_width));
    for (std::uint64_t tile_row = local_start_row; tile_row < local_end_row; tile_row += 1) {
      dequantize_tensor_range_into_f32(
        state,
        tensor,
        (start_row + tile_row) * shape.row_width,
        shape.row_width,
        row_weights);
      std::memcpy(
        tile_weights.data() + static_cast<std::ptrdiff_t>(destination_base + tile_row * shape.row_width),
        row_weights.data(),
        static_cast<std::size_t>(shape.row_width) * sizeof(float));
    }
  };

  const std::uint64_t worker_count = choose_row_worker_count(
    row_count * shape.row_width,
    row_count,
    131072,
    16);
  if (worker_count <= 1) {
    dequantize_rows(0, row_count);
  } else {
    persistent_row_workers().execute(row_count, worker_count, dequantize_rows);
  }
}

static void dequantize_tensor_row_tile_into_f32(
  const LoadedModelState &state,
  const TensorInfo &tensor,
  const MatrixShape &shape,
  std::uint64_t start_row,
  std::uint64_t row_count,
  std::vector<float> &tile_weights) {
  tile_weights.assign(static_cast<std::size_t>(row_count * shape.row_width), 0.0f);
  dequantize_tensor_row_tile_slice_into_f32(
    state,
    tensor,
    shape,
    start_row,
    row_count,
    tile_weights,
    0);
}

static float gelu_tanh(float value);

static bool try_matrix_batch_mul_tensor_resident(
  const LoadedModelState &state,
  const TensorInfo &tensor,
  const MatrixShape &shape,
  const FloatBatch &inputs,
  BatchExecutionConfig *execution_config,
  const std::string &label,
  FloatBatch &outputs) {
  D3d12WeightFormat format = D3d12WeightFormat::F32;
  if (!batch_matmul_gpu_resident_eligible(tensor, shape, inputs, execution_config, &format)) {
    return false;
  }
  const auto started_at = std::chrono::steady_clock::now();
  fill_cached_gpu_packed_inputs(shape, inputs, *execution_config);

  const std::uint8_t *weight_data = tensor_data_ptr(state, tensor);
  D3d12ResidentWeightRef op;
  op.weight_key = weight_data;
  op.format = format;
  op.weight_data = weight_data;
  op.weight_bytes = static_cast<std::size_t>(tensor.byte_length);
  op.row_count = static_cast<std::uint32_t>(shape.row_count);
  op.row_width = static_cast<std::uint32_t>(shape.row_width);

  std::vector<float> flat_outputs;
  std::string gpu_error;
  if (!d3d12_batch_matmul_resident(
    &op,
    1,
    execution_config->cached_gpu_packed_inputs,
    execution_config->cached_gpu_batch_count,
    execution_config->cached_gpu_input_owner,
    &flat_outputs,
    &gpu_error)) {
    execution_config->detail = gpu_error.empty()
      ? "Direct3D 12 resident batched prefill fell back to CPU matmul."
      : gpu_error;
    return false;
  }

  outputs.assign(
    inputs.size(),
    std::vector<float>(static_cast<std::size_t>(shape.row_count), 0.0f));
  for (std::size_t batch_index = 0; batch_index < inputs.size(); batch_index += 1) {
    std::memcpy(
      outputs[batch_index].data(),
      flat_outputs.data() + batch_index * static_cast<std::size_t>(shape.row_count),
      static_cast<std::size_t>(shape.row_count) * sizeof(float));
  }

  execution_config->hybrid_gpu_used = true;
  execution_config->native_backend = d3d12_batch_matmul_capabilities().backend_name;
  if (execution_config->detail.empty()) {
    execution_config->detail = "Direct3D 12 resident quantized batched prefill executed on "
      + d3d12_batch_matmul_capabilities().adapter_name + ".";
  }
  const auto finished_at = std::chrono::steady_clock::now();
  const std::uint64_t elapsed_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count()));
  record_prefill_projection_metric(
    execution_config,
    label,
    shape,
    static_cast<std::uint64_t>(inputs.size()),
    d3d12_batch_matmul_capabilities().backend_name,
    elapsed_ms);
  return true;
}

// Batched q/k/v in one GPU submission: the three projections share the same
// normed layer input, so fusing them removes two ExecuteCommandLists+fence
// round-trips per layer during prefill.
static bool try_fused_qkv_batch_resident(
  const LoadedModelState &state,
  const TensorInfo &query_tensor,
  const MatrixShape &query_shape,
  const TensorInfo &key_tensor,
  const MatrixShape &key_shape,
  const TensorInfo &value_tensor,
  const MatrixShape &value_shape,
  const FloatBatch &inputs,
  BatchExecutionConfig *execution_config,
  FloatBatch &queries,
  FloatBatch &keys,
  FloatBatch &values) {
  D3d12WeightFormat query_format = D3d12WeightFormat::F32;
  D3d12WeightFormat key_format = D3d12WeightFormat::F32;
  D3d12WeightFormat value_format = D3d12WeightFormat::F32;
  if (!gpu_fuse_qkv_enabled()
    || !batch_matmul_gpu_resident_eligible(query_tensor, query_shape, inputs, execution_config, &query_format)
    || key_shape.row_width != query_shape.row_width
    || value_shape.row_width != query_shape.row_width
    || key_shape.row_width * key_shape.row_count != key_tensor.element_count
    || value_shape.row_width * value_shape.row_count != value_tensor.element_count
    || !d3d12_weight_format_supported(key_tensor.ggml_type, key_shape.row_width, &key_format)
    || !d3d12_weight_format_supported(value_tensor.ggml_type, value_shape.row_width, &value_format)) {
    return false;
  }
  const auto started_at = std::chrono::steady_clock::now();
  fill_cached_gpu_packed_inputs(query_shape, inputs, *execution_config);

  const TensorInfo *tensors[3] = {&query_tensor, &key_tensor, &value_tensor};
  const MatrixShape *shapes[3] = {&query_shape, &key_shape, &value_shape};
  const D3d12WeightFormat formats[3] = {query_format, key_format, value_format};
  D3d12ResidentWeightRef ops[3];
  for (std::size_t op_index = 0; op_index < 3; op_index += 1) {
    const std::uint8_t *weight_data = tensor_data_ptr(state, *tensors[op_index]);
    ops[op_index].weight_key = weight_data;
    ops[op_index].format = formats[op_index];
    ops[op_index].weight_data = weight_data;
    ops[op_index].weight_bytes = static_cast<std::size_t>(tensors[op_index]->byte_length);
    ops[op_index].row_count = static_cast<std::uint32_t>(shapes[op_index]->row_count);
    ops[op_index].row_width = static_cast<std::uint32_t>(shapes[op_index]->row_width);
  }

  std::vector<float> flat_outputs[3];
  std::string gpu_error;
  if (!d3d12_batch_matmul_resident(
    ops,
    3,
    execution_config->cached_gpu_packed_inputs,
    execution_config->cached_gpu_batch_count,
    execution_config->cached_gpu_input_owner,
    flat_outputs,
    &gpu_error)) {
    execution_config->detail = gpu_error.empty()
      ? "Direct3D 12 fused QKV prefill fell back to separate projections."
      : gpu_error;
    return false;
  }

  FloatBatch *targets[3] = {&queries, &keys, &values};
  for (std::size_t op_index = 0; op_index < 3; op_index += 1) {
    const std::size_t row_count = static_cast<std::size_t>(shapes[op_index]->row_count);
    targets[op_index]->assign(inputs.size(), std::vector<float>(row_count, 0.0f));
    for (std::size_t batch_index = 0; batch_index < inputs.size(); batch_index += 1) {
      std::memcpy(
        (*targets[op_index])[batch_index].data(),
        flat_outputs[op_index].data() + batch_index * row_count,
        row_count * sizeof(float));
    }
  }

  execution_config->hybrid_gpu_used = true;
  execution_config->native_backend = d3d12_batch_matmul_capabilities().backend_name;
  if (execution_config->detail.empty()) {
    execution_config->detail = "Direct3D 12 resident quantized batched prefill executed on "
      + d3d12_batch_matmul_capabilities().adapter_name + ".";
  }
  const auto finished_at = std::chrono::steady_clock::now();
  const std::uint64_t elapsed_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count()));
  record_prefill_projection_metric(
    execution_config,
    "Gemma batched attention QKV projection",
    query_shape,
    static_cast<std::uint64_t>(inputs.size()),
    d3d12_batch_matmul_capabilities().backend_name,
    elapsed_ms);
  return true;
}

// Whole batched-prefill FFN block in one GPU submission; see the decode
// variant above. Records a single projection metric for the chain.
static bool try_ffn_chain_batch_resident(
  const LoadedModelState &state,
  const TensorInfo &gate_tensor,
  const MatrixShape &gate_shape,
  const TensorInfo &up_tensor,
  const MatrixShape &up_shape,
  const TensorInfo &down_tensor,
  const MatrixShape &down_shape,
  const FloatBatch &inputs,
  BatchExecutionConfig *execution_config,
  FloatBatch &outputs,
  bool use_silu = false) {
  D3d12WeightFormat gate_format = D3d12WeightFormat::F32;
  if (!gpu_fuse_ffn_enabled()
    || !batch_matmul_gpu_resident_eligible(gate_tensor, gate_shape, inputs, execution_config, &gate_format)) {
    return false;
  }
  D3d12ResidentWeightRef gate_ref;
  D3d12ResidentWeightRef up_ref;
  D3d12ResidentWeightRef down_ref;
  if (!build_resident_ref(state, gate_tensor, gate_shape, gate_ref)
    || !build_resident_ref(state, up_tensor, up_shape, up_ref)
    || !build_resident_ref(state, down_tensor, down_shape, down_ref)) {
    return false;
  }
  const auto started_at = std::chrono::steady_clock::now();
  fill_cached_gpu_packed_inputs(gate_shape, inputs, *execution_config);

  std::vector<float> flat_outputs;
  std::string gpu_error;
  const bool batch_chain_ok = use_silu
    ? d3d12_ffn_silu_glu_resident(
      gate_ref,
      up_ref,
      down_ref,
      execution_config->cached_gpu_packed_inputs,
      execution_config->cached_gpu_batch_count,
      execution_config->cached_gpu_input_owner,
      &flat_outputs,
      &gpu_error)
    : d3d12_ffn_gelu_glu_resident(
      gate_ref,
      up_ref,
      down_ref,
      execution_config->cached_gpu_packed_inputs,
      execution_config->cached_gpu_batch_count,
      execution_config->cached_gpu_input_owner,
      &flat_outputs,
      &gpu_error);
  if (!batch_chain_ok) {
    execution_config->detail = gpu_error.empty()
      ? "Direct3D 12 fused FFN chain fell back to separate projections."
      : gpu_error;
    return false;
  }

  // Diagnostic guard: a non-finite chain output means the fused GPU path
  // disagreed with the model state; log it and use the unfused path instead.
  for (std::size_t index = 0; index < flat_outputs.size(); index += 1) {
    if (!std::isfinite(flat_outputs[index])) {
      std::size_t finite_count = 0;
      for (const float value : flat_outputs) {
        if (std::isfinite(value)) {
          finite_count += 1;
        }
      }
      std::ostringstream diagnostics;
      diagnostics << "Fused FFN chain non-finite: layer=" << execution_config->current_layer_index
        << " firstBadIndex=" << index
        << " (batch " << index / static_cast<std::size_t>(down_shape.row_count)
        << ", row " << index % static_cast<std::size_t>(down_shape.row_count) << ")"
        << " finite=" << finite_count << "/" << flat_outputs.size()
        << " v0=" << flat_outputs[0]
        << " v1=" << flat_outputs[1]
        << " badPrev=" << (index > 0 ? flat_outputs[index - 1] : 0.0f);
      execution_config->detail = diagnostics.str();
      return false;
    }
  }

  const std::size_t down_rows = static_cast<std::size_t>(down_shape.row_count);
  outputs.assign(inputs.size(), std::vector<float>(down_rows, 0.0f));
  for (std::size_t batch_index = 0; batch_index < inputs.size(); batch_index += 1) {
    std::memcpy(
      outputs[batch_index].data(),
      flat_outputs.data() + batch_index * down_rows,
      down_rows * sizeof(float));
  }

  execution_config->hybrid_gpu_used = true;
  execution_config->native_backend = d3d12_batch_matmul_capabilities().backend_name;
  if (execution_config->detail.empty()) {
    execution_config->detail = "Direct3D 12 resident quantized batched prefill executed on "
      + d3d12_batch_matmul_capabilities().adapter_name + ".";
  }
  const auto finished_at = std::chrono::steady_clock::now();
  const std::uint64_t elapsed_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count()));
  record_prefill_projection_metric(
    execution_config,
    "Gemma batched MLP FFN chain",
    gate_shape,
    static_cast<std::uint64_t>(inputs.size()),
    d3d12_batch_matmul_capabilities().backend_name,
    elapsed_ms);
  return true;
}

static FloatBatch matrix_batch_mul_tensor(
  const LoadedModelState &state,
  const TensorInfo &tensor,
  const MatrixShape &shape,
  const FloatBatch &inputs,
  BatchExecutionConfig *execution_config,
  const std::string &label) {
  const auto started_at = std::chrono::steady_clock::now();
  if (inputs.empty()) {
    return {};
  }
  for (const std::vector<float> &input : inputs) {
    if (shape.row_width != input.size()) {
      throw WorkerError("NATIVE_WORKER_ERROR", label + " width does not match a batched input vector.");
    }
  }
  if (shape.row_width == 0
    || shape.row_count == 0
    || shape.row_count > (std::numeric_limits<std::uint64_t>::max() / shape.row_width)
    || shape.row_width * shape.row_count != tensor.element_count) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " has an invalid tensor shape.");
  }

  {
    FloatBatch resident_outputs;
    if (try_matrix_batch_mul_tensor_resident(
      state,
      tensor,
      shape,
      inputs,
      execution_config,
      label,
      resident_outputs)) {
      return resident_outputs;
    }
  }

  if (batch_matmul_gpu_eligible(shape, inputs, execution_config)) {
    const std::uint64_t tile_rows = std::max<std::uint64_t>(16, execution_config->gpu_tile_rows);
    fill_cached_gpu_packed_inputs(shape, inputs, *execution_config);
    FloatBatch outputs(
      inputs.size(),
      std::vector<float>(static_cast<std::size_t>(shape.row_count), 0.0f));
    std::vector<float> tile_weights;
    std::string gpu_error;
    bool used_gpu = false;
    bool cpu_remainder = false;
    for (std::uint64_t row = 0; row < shape.row_count; row += tile_rows) {
      const std::uint64_t current_tile_rows = std::min<std::uint64_t>(tile_rows, shape.row_count - row);
      dequantize_tensor_row_tile_into_f32(
        state,
        tensor,
        shape,
        row,
        current_tile_rows,
        tile_weights);

      if (!cpu_remainder) {
        FloatBatch tile_outputs;
        const bool tile_success = d3d12_batch_matmul(
          tile_weights,
          static_cast<std::uint32_t>(current_tile_rows),
          static_cast<std::uint32_t>(shape.row_width),
          execution_config->cached_gpu_packed_inputs,
          execution_config->cached_gpu_batch_count,
          execution_config->cached_gpu_input_owner,
          tile_outputs,
          &gpu_error);
        if (!tile_success) {
          execution_config->detail = gpu_error.empty()
            ? "Direct3D 12 hybrid prefill fell back to CPU matmul."
            : gpu_error;
          if (!used_gpu) {
            break;
          }
          cpu_remainder = true;
        } else {
          used_gpu = true;
          for (std::size_t batch_index = 0; batch_index < tile_outputs.size(); batch_index += 1) {
            std::copy(
              tile_outputs[batch_index].begin(),
              tile_outputs[batch_index].end(),
              outputs[batch_index].begin() + static_cast<std::ptrdiff_t>(row));
          }
          continue;
        }
      }

      for (std::uint64_t tile_row = 0; tile_row < current_tile_rows; tile_row += 1) {
        const float *row_data = tile_weights.data() + static_cast<std::ptrdiff_t>(tile_row * shape.row_width);
        for (std::size_t batch_index = 0; batch_index < inputs.size(); batch_index += 1) {
          float dot = 0.0f;
          for (std::size_t item = 0; item < inputs[batch_index].size(); item += 1) {
            dot += row_data[item] * inputs[batch_index][item];
          }
          outputs[batch_index][static_cast<std::size_t>(row + tile_row)] = dot;
        }
      }
    }
    if (used_gpu) {
      execution_config->hybrid_gpu_used = true;
      execution_config->native_backend = d3d12_batch_matmul_capabilities().backend_name;
      if (execution_config->detail.empty()) {
        execution_config->detail = "Direct3D 12 hybrid batched prefill executed on "
          + d3d12_batch_matmul_capabilities().adapter_name + ".";
      }
      const auto finished_at = std::chrono::steady_clock::now();
      const std::uint64_t elapsed_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
        1,
        std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count()));
      record_prefill_projection_metric(
        execution_config,
        label,
        shape,
        static_cast<std::uint64_t>(inputs.size()),
        cpu_remainder
          ? d3d12_batch_matmul_capabilities().backend_name + "+cpu"
          : d3d12_batch_matmul_capabilities().backend_name,
        elapsed_ms);
      return outputs;
    }
  }

  FloatBatch outputs(
    inputs.size(),
    std::vector<float>(static_cast<std::size_t>(shape.row_count), 0.0f));
  const std::uint64_t total_work = shape.row_count * shape.row_width * inputs.size();
  const std::uint64_t worker_count = choose_row_worker_count(total_work, shape.row_count);
  const auto compute_rows = [&](std::uint64_t start_row, std::uint64_t end_row) {
    std::vector<float> row_weights;
    row_weights.reserve(static_cast<std::size_t>(shape.row_width));
    for (std::uint64_t row = start_row; row < end_row; row += 1) {
      dequantize_tensor_range_into_f32(
        state,
        tensor,
        row * shape.row_width,
        shape.row_width,
        row_weights);
      for (std::size_t batch_index = 0; batch_index < inputs.size(); batch_index += 1) {
        float dot = 0.0f;
        const std::vector<float> &input = inputs[batch_index];
        for (std::size_t item = 0; item < row_weights.size(); item += 1) {
          dot += row_weights[item] * input[item];
        }
        outputs[batch_index][static_cast<std::size_t>(row)] = dot;
      }
    }
  };

  if (worker_count <= 1) {
    compute_rows(0, shape.row_count);
  } else {
    persistent_row_workers().execute(shape.row_count, worker_count, compute_rows);
  }
  const auto finished_at = std::chrono::steady_clock::now();
  const std::uint64_t elapsed_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count()));
  record_prefill_projection_metric(
    execution_config,
    label,
    shape,
    static_cast<std::uint64_t>(inputs.size()),
    "cpu",
    elapsed_ms);
  return outputs;
}

static bool try_fused_gelu_glu_projection_tensor_batch_hybrid(
  const LoadedModelState &state,
  const TensorInfo &gate_tensor,
  const MatrixShape &gate_shape,
  const TensorInfo &up_tensor,
  const MatrixShape &up_shape,
  const FloatBatch &inputs,
  BatchExecutionConfig *execution_config,
  const std::string &label,
  FloatBatch &outputs) {
  const auto started_at = std::chrono::steady_clock::now();
  if (gate_shape.row_count != up_shape.row_count
    || gate_shape.row_width != up_shape.row_width
    || gate_shape.row_count > (std::numeric_limits<std::uint32_t>::max() / 2)) {
    return false;
  }

  D3d12WeightFormat gate_format = D3d12WeightFormat::F32;
  D3d12WeightFormat up_format = D3d12WeightFormat::F32;
  if (batch_matmul_gpu_resident_eligible(gate_tensor, gate_shape, inputs, execution_config, &gate_format)
    && d3d12_weight_format_supported(up_tensor.ggml_type, up_shape.row_width, &up_format)) {
    fill_cached_gpu_packed_inputs(gate_shape, inputs, *execution_config);
    const std::uint8_t *gate_data = tensor_data_ptr(state, gate_tensor);
    const std::uint8_t *up_data = tensor_data_ptr(state, up_tensor);
    D3d12ResidentWeightRef ops[2];
    ops[0].weight_key = gate_data;
    ops[0].format = gate_format;
    ops[0].weight_data = gate_data;
    ops[0].weight_bytes = static_cast<std::size_t>(gate_tensor.byte_length);
    ops[0].row_count = static_cast<std::uint32_t>(gate_shape.row_count);
    ops[0].row_width = static_cast<std::uint32_t>(gate_shape.row_width);
    ops[1].weight_key = up_data;
    ops[1].format = up_format;
    ops[1].weight_data = up_data;
    ops[1].weight_bytes = static_cast<std::size_t>(up_tensor.byte_length);
    ops[1].row_count = static_cast<std::uint32_t>(up_shape.row_count);
    ops[1].row_width = static_cast<std::uint32_t>(up_shape.row_width);

    std::vector<float> flat_outputs[2];
    std::string gpu_error;
    if (d3d12_batch_matmul_resident(
      ops,
      2,
      execution_config->cached_gpu_packed_inputs,
      execution_config->cached_gpu_batch_count,
      execution_config->cached_gpu_input_owner,
      flat_outputs,
      &gpu_error)) {
      const std::size_t row_count = static_cast<std::size_t>(gate_shape.row_count);
      outputs.assign(inputs.size(), std::vector<float>(row_count, 0.0f));
      for (std::size_t batch_index = 0; batch_index < inputs.size(); batch_index += 1) {
        const float *gate_values = flat_outputs[0].data() + batch_index * row_count;
        const float *up_values = flat_outputs[1].data() + batch_index * row_count;
        std::vector<float> &fused = outputs[batch_index];
        for (std::size_t row = 0; row < row_count; row += 1) {
          fused[row] = gelu_tanh(gate_values[row]) * up_values[row];
        }
      }
      execution_config->hybrid_gpu_used = true;
      execution_config->native_backend = d3d12_batch_matmul_capabilities().backend_name;
      if (execution_config->detail.empty()) {
        execution_config->detail = "Direct3D 12 resident fused GELU-GLU prefill executed on "
          + d3d12_batch_matmul_capabilities().adapter_name + ".";
      }
      const auto resident_finished_at = std::chrono::steady_clock::now();
      const std::uint64_t resident_elapsed_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
        1,
        std::chrono::duration_cast<std::chrono::milliseconds>(resident_finished_at - started_at).count()));
      record_prefill_projection_metric(
        execution_config,
        label,
        gate_shape,
        static_cast<std::uint64_t>(inputs.size()),
        d3d12_batch_matmul_capabilities().backend_name,
        resident_elapsed_ms);
      return true;
    }
    execution_config->detail = gpu_error.empty()
      ? "Direct3D 12 resident fused GELU-GLU prefill fell back to the tiled hybrid path."
      : gpu_error;
  }

  if (!batch_matmul_gpu_eligible(gate_shape, inputs, execution_config)) {
    return false;
  }

  const std::uint64_t tile_rows = std::max<std::uint64_t>(16, execution_config->gpu_tile_rows);
  fill_cached_gpu_packed_inputs(gate_shape, inputs, *execution_config);
  outputs.assign(
    inputs.size(),
    std::vector<float>(static_cast<std::size_t>(gate_shape.row_count), 0.0f));

  std::vector<float> tile_weights;
  std::string gpu_error;
  bool used_gpu = false;
  bool cpu_remainder = false;
  for (std::uint64_t row = 0; row < gate_shape.row_count; row += tile_rows) {
    const std::uint64_t current_tile_rows = std::min<std::uint64_t>(tile_rows, gate_shape.row_count - row);
    tile_weights.assign(static_cast<std::size_t>(current_tile_rows * gate_shape.row_width * 2), 0.0f);
    dequantize_tensor_row_tile_slice_into_f32(
      state,
      gate_tensor,
      gate_shape,
      row,
      current_tile_rows,
      tile_weights,
      0);
    dequantize_tensor_row_tile_slice_into_f32(
      state,
      up_tensor,
      up_shape,
      row,
      current_tile_rows,
      tile_weights,
      current_tile_rows);

    if (!cpu_remainder) {
      FloatBatch tile_outputs;
      const bool tile_success = d3d12_batch_matmul(
        tile_weights,
        static_cast<std::uint32_t>(current_tile_rows * 2),
        static_cast<std::uint32_t>(gate_shape.row_width),
        execution_config->cached_gpu_packed_inputs,
        execution_config->cached_gpu_batch_count,
        execution_config->cached_gpu_input_owner,
        tile_outputs,
        &gpu_error);
      if (!tile_success) {
        execution_config->detail = gpu_error.empty()
          ? "Direct3D 12 hybrid fused GELU-GLU prefill fell back to CPU matmul."
          : gpu_error;
        if (!used_gpu) {
          return false;
        }
        cpu_remainder = true;
      } else {
        used_gpu = true;
        for (std::size_t batch_index = 0; batch_index < tile_outputs.size(); batch_index += 1) {
          for (std::size_t tile_row = 0; tile_row < static_cast<std::size_t>(current_tile_rows); tile_row += 1) {
            const float gate_value = tile_outputs[batch_index][tile_row];
            const float up_value = tile_outputs[batch_index][tile_row + static_cast<std::size_t>(current_tile_rows)];
            outputs[batch_index][static_cast<std::size_t>(row) + tile_row] = gelu_tanh(gate_value) * up_value;
          }
        }
        continue;
      }
    }

    for (std::uint64_t tile_row = 0; tile_row < current_tile_rows; tile_row += 1) {
      const float *gate_row_data = tile_weights.data() + static_cast<std::ptrdiff_t>(tile_row * gate_shape.row_width);
      const float *up_row_data = tile_weights.data()
        + static_cast<std::ptrdiff_t>((current_tile_rows + tile_row) * gate_shape.row_width);
      for (std::size_t batch_index = 0; batch_index < inputs.size(); batch_index += 1) {
        float gate_dot = 0.0f;
        float up_dot = 0.0f;
        for (std::size_t item = 0; item < inputs[batch_index].size(); item += 1) {
          const float input_value = inputs[batch_index][item];
          gate_dot += gate_row_data[item] * input_value;
          up_dot += up_row_data[item] * input_value;
        }
        outputs[batch_index][static_cast<std::size_t>(row + tile_row)] = gelu_tanh(gate_dot) * up_dot;
      }
    }
  }

  if (!used_gpu) {
    return false;
  }

  execution_config->hybrid_gpu_used = true;
  execution_config->native_backend = d3d12_batch_matmul_capabilities().backend_name;
  if (execution_config->detail.empty()) {
    execution_config->detail = "Direct3D 12 hybrid fused GELU-GLU prefill executed on "
      + d3d12_batch_matmul_capabilities().adapter_name + ".";
  }
  const auto finished_at = std::chrono::steady_clock::now();
  const std::uint64_t elapsed_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count()));
  record_prefill_projection_metric(
    execution_config,
    label,
    gate_shape,
    static_cast<std::uint64_t>(inputs.size()),
    cpu_remainder
      ? d3d12_batch_matmul_capabilities().backend_name + "+cpu"
      : d3d12_batch_matmul_capabilities().backend_name,
    elapsed_ms);
  return true;
}

static float gelu_tanh(float value) {
  constexpr float gelu_scale = 0.7978845608028654f;
  const float value_cubed = value * value * value;
  return 0.5f * value
    * (1.0f + std::tanh(gelu_scale * (value + 0.044715f * value_cubed)));
}

static std::vector<float> fused_gelu_glu_projection_tensor(
  const LoadedModelState &state,
  const TensorInfo &gate_tensor,
  const MatrixShape &gate_shape,
  const TensorInfo &up_tensor,
  const MatrixShape &up_shape,
  const std::vector<float> &input,
  const std::string &label) {
  if (gate_shape.row_width != input.size() || up_shape.row_width != input.size()) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " width does not match the input vector size.");
  }
  if (gate_shape.row_count != up_shape.row_count) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " gate and up projections must have the same row count.");
  }
  if (gate_shape.row_width == 0
    || gate_shape.row_count == 0
    || gate_shape.row_count > (std::numeric_limits<std::uint64_t>::max() / gate_shape.row_width)
    || up_shape.row_count > (std::numeric_limits<std::uint64_t>::max() / up_shape.row_width)) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " has an invalid tensor shape.");
  }
  if (gate_shape.row_width * gate_shape.row_count != gate_tensor.element_count
    || up_shape.row_width * up_shape.row_count != up_tensor.element_count) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " tensor payload does not match the advertised tensor shape.");
  }

  // Decode-time GPU lane for the fused FFN pair: both matvecs share one
  // submission against resident weights; the GELU-GLU combine stays on CPU.
  if (gpu_decode_lane_enabled_for_model(state)
    && gate_shape.row_count * gate_shape.row_width >= 4194304) {
    D3d12WeightFormat gate_format = D3d12WeightFormat::F32;
    D3d12WeightFormat up_format = D3d12WeightFormat::F32;
    if (d3d12_weight_format_supported(gate_tensor.ggml_type, gate_shape.row_width, &gate_format)
      && d3d12_weight_format_supported(up_tensor.ggml_type, up_shape.row_width, &up_format)) {
      const std::uint8_t *gate_data = tensor_data_ptr(state, gate_tensor);
      const std::uint8_t *up_data = tensor_data_ptr(state, up_tensor);
      D3d12ResidentWeightRef ops[2];
      ops[0].weight_key = gate_data;
      ops[0].format = gate_format;
      ops[0].weight_data = gate_data;
      ops[0].weight_bytes = static_cast<std::size_t>(gate_tensor.byte_length);
      ops[0].row_count = static_cast<std::uint32_t>(gate_shape.row_count);
      ops[0].row_width = static_cast<std::uint32_t>(gate_shape.row_width);
      ops[1].weight_key = up_data;
      ops[1].format = up_format;
      ops[1].weight_data = up_data;
      ops[1].weight_bytes = static_cast<std::size_t>(up_tensor.byte_length);
      ops[1].row_count = static_cast<std::uint32_t>(up_shape.row_count);
      ops[1].row_width = static_cast<std::uint32_t>(up_shape.row_width);
      std::vector<float> flat_outputs[2];
      if (d3d12_batch_matmul_resident(
        ops, 2, input, 1, next_gpu_input_identity(), flat_outputs, nullptr)) {
        mark_decode_gpu_used(
          "Direct3D 12 resident decode GELU-GLU pair executed on "
          + d3d12_batch_matmul_capabilities().adapter_name + ".");
        std::vector<float> fused(static_cast<std::size_t>(gate_shape.row_count), 0.0f);
        for (std::size_t row = 0; row < fused.size(); row += 1) {
          fused[row] = gelu_tanh(flat_outputs[0][row]) * flat_outputs[1][row];
        }
        return fused;
      }
    }
  }

  std::vector<float> output(static_cast<std::size_t>(gate_shape.row_count), 0.0f);
  const std::uint64_t total_work = gate_shape.row_count * gate_shape.row_width * 2;
  const std::uint64_t worker_count = total_work >= 262144
    ? std::min<std::uint64_t>(
      static_cast<std::uint64_t>(max_cpu_row_workers()),
      std::max<std::uint64_t>(1, gate_shape.row_count / 64))
    : 1;
  const std::string gate_label = label + " gate";
  const std::string up_label = label + " up";
  const auto compute_rows = [&](std::uint64_t start_row, std::uint64_t end_row) {
    for (std::uint64_t row = start_row; row < end_row; row += 1) {
      double gate_dot = 0.0;
      double up_dot = 0.0;
      if (!tensor_range_dual_dot_f32(
        state,
        gate_tensor,
        up_tensor,
        row * gate_shape.row_width,
        gate_shape.row_width,
        input,
        &gate_dot,
        &up_dot)) {
        gate_dot = tensor_range_dot_f32(
          state,
          gate_tensor,
          row * gate_shape.row_width,
          gate_shape.row_width,
          input,
          gate_label);
        up_dot = tensor_range_dot_f32(
          state,
          up_tensor,
          row * up_shape.row_width,
          up_shape.row_width,
          input,
          up_label);
      }
      const float gate = static_cast<float>(gate_dot);
      const float up = static_cast<float>(up_dot);
      output[static_cast<std::size_t>(row)] = gelu_tanh(gate) * up;
    }
  };

  if (worker_count <= 1) {
    compute_rows(0, gate_shape.row_count);
    return output;
  }

  persistent_row_workers().execute(gate_shape.row_count, worker_count, compute_rows);
  return output;
}

static FloatBatch fused_gelu_glu_projection_tensor_batch(
  const LoadedModelState &state,
  const TensorInfo &gate_tensor,
  const MatrixShape &gate_shape,
  const TensorInfo &up_tensor,
  const MatrixShape &up_shape,
  const FloatBatch &inputs,
  BatchExecutionConfig *execution_config,
  const std::string &label) {
  if (gate_shape.row_width != up_shape.row_width) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " gate and up projections must have the same row width.");
  }
  if (gate_shape.row_count != up_shape.row_count) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " gate and up projections must have the same row count.");
  }

  FloatBatch fused_outputs;
  if (try_fused_gelu_glu_projection_tensor_batch_hybrid(
    state,
    gate_tensor,
    gate_shape,
    up_tensor,
    up_shape,
    inputs,
    execution_config,
    label,
    fused_outputs)) {
    return fused_outputs;
  }

  FloatBatch gate = matrix_batch_mul_tensor(
    state,
    gate_tensor,
    gate_shape,
    inputs,
    execution_config,
    label + " gate");
  FloatBatch up = matrix_batch_mul_tensor(
    state,
    up_tensor,
    up_shape,
    inputs,
    execution_config,
    label + " up");
  if (gate.size() != up.size()) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " batched projections produced mismatched batch sizes.");
  }
  for (std::size_t batch_index = 0; batch_index < gate.size(); batch_index += 1) {
    if (gate[batch_index].size() != up[batch_index].size()) {
      throw WorkerError("NATIVE_WORKER_ERROR", label + " batched projections produced mismatched shapes.");
    }
    for (std::size_t item = 0; item < gate[batch_index].size(); item += 1) {
      gate[batch_index][item] = gelu_tanh(gate[batch_index][item]) * up[batch_index][item];
    }
  }
  return gate;
}

static bool tensor_type_supported_for_execution(const TensorInfo &tensor) {
  return tensor.ggml_type == 0
    || tensor.ggml_type == 1
    || tensor.ggml_type == 2
    || tensor.ggml_type == 8
    || tensor.ggml_type == 10
    || tensor.ggml_type == 11
    || tensor.ggml_type == 12
    || tensor.ggml_type == 13
    || tensor.ggml_type == 14
    || tensor.ggml_type == 23
    || tensor.ggml_type == 30
    || tensor.ggml_type == 34
    || tensor.ggml_type == 35;
}

static std::optional<LlamaExecutionConfig> resolve_llama_execution_config(const LoadedModelState &state) {
  if (!state.model.embedding_length.has_value() || !state.model.attention_head_count.has_value()) {
    return std::nullopt;
  }
  const std::uint64_t hidden_size = state.model.embedding_length.value();
  const std::uint64_t head_count = state.model.attention_head_count.value();
  const std::uint64_t kv_head_count = state.model.kv_head_count.value_or(head_count);
  if (hidden_size == 0 || head_count == 0 || kv_head_count == 0) {
    return std::nullopt;
  }
  if (hidden_size % head_count != 0 || head_count % kv_head_count != 0) {
    return std::nullopt;
  }
  const std::uint64_t head_dim = hidden_size / head_count;
  if (head_dim == 0 || (head_dim % 2) != 0) {
    return std::nullopt;
  }
  std::vector<float> rope_freq_factors;
  if (state.tensor_index.rope_freqs != k_missing_tensor_index) {
    const TensorInfo &rope_tensor = tensor_at(state, state.tensor_index.rope_freqs);
    if (rope_tensor.dimensions.size() != 1) {
      return std::nullopt;
    }
    rope_freq_factors = cached_tensor_vector_f32(state, state.tensor_index.rope_freqs, "Llama RoPE frequency factors");
  }
  return LlamaExecutionConfig {
    hidden_size,
    head_count,
    kv_head_count,
    head_dim,
    state.model.rope_freq_base.value_or(10000.0),
    state.model.rope_freq_scale.value_or(1.0),
    rope_freq_factors,
  };
}

// Cached form of resolve_llama_execution_config. The uncached resolver runs
// once per consume-token call and copies rope_freq_factors each time; the
// model metadata it reads never changes after load, so resolve once and
// hand out a stable pointer (nullptr when the model is not llama-executable).
static const LlamaExecutionConfig *cached_llama_execution_config(const LoadedModelState &state) {
  if (!state.llama_execution_config_resolved) {
    state.llama_execution_config_cache = resolve_llama_execution_config(state);
    state.llama_execution_config_resolved = true;
  }
  return state.llama_execution_config_cache.has_value()
    ? &state.llama_execution_config_cache.value()
    : nullptr;
}

struct FamilyExecutionConfig {
  FamilySpec spec;
  std::uint64_t hidden_size = 0;
  std::uint64_t head_count = 0;
  std::uint64_t kv_head_count = 0;
  std::uint64_t head_dim = 0;
  std::uint64_t rope_dim = 0;
  double rope_freq_base = 10000.0;
  double rope_freq_scale = 1.0;
  double rope_attn_factor = 1.0;
  std::vector<float> rope_freq_factors;
  double norm_epsilon = 1e-5;
  std::uint64_t expert_count = 0;
  std::uint64_t expert_used_count = 0;
  double expert_weights_scale = 1.0;
};

static std::optional<FamilyExecutionConfig> resolve_family_execution_config(const LoadedModelState &state) {
  const auto spec = resolve_family_spec(state.model);
  if (!spec.has_value()) {
    return std::nullopt;
  }
  if (!state.model.embedding_length.has_value() || !state.model.attention_head_count.has_value()) {
    return std::nullopt;
  }
  const std::uint64_t hidden_size = state.model.embedding_length.value();
  const std::uint64_t head_count = state.model.attention_head_count.value();
  const std::uint64_t kv_head_count = state.model.kv_head_count.value_or(head_count);
  if (hidden_size == 0 || head_count == 0 || kv_head_count == 0 || head_count % kv_head_count != 0) {
    return std::nullopt;
  }
  std::uint64_t head_dim = state.model.attention_key_length.value_or(0);
  if (head_dim == 0) {
    if (hidden_size % head_count != 0) {
      return std::nullopt;
    }
    head_dim = hidden_size / head_count;
  }
  if (head_dim == 0 || head_dim % 2 != 0) {
    return std::nullopt;
  }
  std::uint64_t rope_dim = head_dim;
  if (spec->rope_partial) {
    rope_dim = state.model.rope_dimension_count.value_or(head_dim);
  }
  if (rope_dim == 0 || rope_dim > head_dim || rope_dim % 2 != 0) {
    return std::nullopt;
  }

  FamilyExecutionConfig config;
  config.spec = spec.value();
  config.hidden_size = hidden_size;
  config.head_count = head_count;
  config.kv_head_count = kv_head_count;
  config.head_dim = head_dim;
  config.rope_dim = rope_dim;
  config.rope_freq_base = state.model.rope_freq_base.value_or(10000.0);
  config.rope_freq_scale = state.model.rope_freq_scale.value_or(1.0);
  config.norm_epsilon = config.spec.norm_kind == FamilyNormKind::layer
    ? state.model.layer_norm_epsilon.value_or(1e-5)
    : state.model.layer_norm_rms_epsilon.value_or(1e-5);
  config.expert_count = state.model.expert_count.value_or(0);
  config.expert_used_count = state.model.expert_used_count.value_or(0);
  config.expert_weights_scale = state.model.expert_weights_scale.value_or(1.0);
  if (config.spec.moe && (config.expert_count == 0 || config.expert_used_count == 0)) {
    return std::nullopt;
  }

  if (config.spec.rope_long_short_factors) {
    // Worker sessions run at or below the model's original training context,
    // matching llama.cpp's factor selection for n_ctx <= original context;
    // the long factors are only correct past that boundary.
    const std::size_t factors_index = state.tensor_index.rope_factors_short != k_missing_tensor_index
      ? state.tensor_index.rope_factors_short
      : state.tensor_index.rope_factors_long;
    if (factors_index != k_missing_tensor_index) {
      const TensorInfo &factors_tensor = tensor_at(state, factors_index);
      if (factors_tensor.dimensions.size() != 1) {
        return std::nullopt;
      }
      config.rope_freq_factors = cached_tensor_vector_f32(state, factors_index, "Family RoPE frequency factors");
      config.rope_attn_factor = state.model.rope_scaling_attn_factor.value_or(1.0);
    }
  } else if (state.tensor_index.rope_freqs != k_missing_tensor_index) {
    const TensorInfo &rope_tensor = tensor_at(state, state.tensor_index.rope_freqs);
    if (rope_tensor.dimensions.size() != 1) {
      return std::nullopt;
    }
    config.rope_freq_factors = cached_tensor_vector_f32(state, state.tensor_index.rope_freqs, "Family RoPE frequency factors");
  }
  return config;
}

static std::optional<GemmaLayerExecutionConfig> resolve_gemma_layer_execution_config(
  const LoadedModelState &state,
  const LlamaLayerTensorIndex &layer_index) {
  if (!state.model.embedding_length.has_value() || !state.model.attention_head_count.has_value()) {
    return std::nullopt;
  }
  const std::uint64_t hidden_size = state.model.embedding_length.value();
  const std::uint64_t head_count = state.model.attention_head_count.value();
  const std::uint64_t kv_head_count = state.model.kv_head_count.value_or(head_count);
  if (hidden_size == 0 || head_count == 0 || kv_head_count == 0 || head_count % kv_head_count != 0) {
    return std::nullopt;
  }
  if (layer_index.query == k_missing_tensor_index
    || layer_index.key == k_missing_tensor_index
    || layer_index.value == k_missing_tensor_index
    || layer_index.output == k_missing_tensor_index
    || layer_index.gate == k_missing_tensor_index
    || layer_index.down == k_missing_tensor_index
    || layer_index.up == k_missing_tensor_index) {
    return std::nullopt;
  }

  const MatrixShape query_shape = tensor_matrix_shape(tensor_at(state, layer_index.query), "Gemma attention query projection");
  const MatrixShape key_shape = tensor_matrix_shape(tensor_at(state, layer_index.key), "Gemma attention key projection");
  const MatrixShape value_shape = tensor_matrix_shape(tensor_at(state, layer_index.value), "Gemma attention value projection");
  const MatrixShape output_shape = tensor_matrix_shape(tensor_at(state, layer_index.output), "Gemma attention output projection");
  const MatrixShape gate_shape = tensor_matrix_shape(tensor_at(state, layer_index.gate), "Gemma MLP gate projection");
  const MatrixShape down_shape = tensor_matrix_shape(tensor_at(state, layer_index.down), "Gemma MLP down projection");
  const MatrixShape up_shape = tensor_matrix_shape(tensor_at(state, layer_index.up), "Gemma MLP up projection");
  const std::uint64_t per_layer_input_size = state.model.embedding_length_per_layer_input.value_or(0);
  MatrixShape per_layer_input_gate_shape;
  MatrixShape per_layer_projection_shape;
  if (per_layer_input_size > 0) {
    if (layer_index.per_layer_input_gate == k_missing_tensor_index
      || layer_index.per_layer_projection == k_missing_tensor_index
      || layer_index.post_norm == k_missing_tensor_index
      || layer_index.layer_output_scale == k_missing_tensor_index) {
      return std::nullopt;
    }
    per_layer_input_gate_shape = tensor_matrix_shape(
      tensor_at(state, layer_index.per_layer_input_gate),
      "Gemma per-layer input gate");
    per_layer_projection_shape = tensor_matrix_shape(
      tensor_at(state, layer_index.per_layer_projection),
      "Gemma per-layer projection");
    const TensorInfo &per_layer_post_norm = tensor_at(state, layer_index.post_norm);
    const TensorInfo &layer_output_scale = tensor_at(state, layer_index.layer_output_scale);
    if (per_layer_input_gate_shape.row_width != hidden_size
      || per_layer_input_gate_shape.row_count != per_layer_input_size
      || per_layer_projection_shape.row_width != per_layer_input_size
      || per_layer_projection_shape.row_count != hidden_size
      || per_layer_post_norm.dimensions.size() != 1
      || per_layer_post_norm.element_count != hidden_size
      || layer_output_scale.dimensions.size() != 1
      || layer_output_scale.element_count != 1) {
      return std::nullopt;
    }
  }
  if (query_shape.row_width != hidden_size
    || key_shape.row_width != hidden_size
    || value_shape.row_width != hidden_size
    || output_shape.row_count != hidden_size
    || gate_shape.row_width != hidden_size
    || up_shape.row_width != hidden_size
    || gate_shape.row_count != up_shape.row_count
    || down_shape.row_width != gate_shape.row_count
    || down_shape.row_count != hidden_size
    || query_shape.row_count == 0
    || key_shape.row_count == 0
    || value_shape.row_count == 0
    || gate_shape.row_count == 0
    || output_shape.row_width != query_shape.row_count
    || query_shape.row_count % head_count != 0
    || key_shape.row_count % kv_head_count != 0
    || value_shape.row_count != key_shape.row_count) {
    return std::nullopt;
  }
  const std::uint64_t head_dim = query_shape.row_count / head_count;
  if (head_dim == 0 || (head_dim % 2) != 0 || key_shape.row_count / kv_head_count != head_dim) {
    return std::nullopt;
  }
  if (layer_index.query_norm == k_missing_tensor_index || layer_index.key_norm == k_missing_tensor_index) {
    return std::nullopt;
  }
  const TensorInfo &query_norm_tensor = tensor_at(state, layer_index.query_norm);
  const TensorInfo &key_norm_tensor = tensor_at(state, layer_index.key_norm);
  if (query_norm_tensor.dimensions.size() != 1
    || key_norm_tensor.dimensions.size() != 1
    || query_norm_tensor.element_count != head_dim
    || key_norm_tensor.element_count != head_dim) {
    return std::nullopt;
  }

  std::vector<float> rope_freq_factors;
  if (state.tensor_index.rope_freqs != k_missing_tensor_index) {
    const TensorInfo &rope_tensor = tensor_at(state, state.tensor_index.rope_freqs);
    if (rope_tensor.dimensions.size() != 1) {
      return std::nullopt;
    }
    rope_freq_factors = cached_tensor_vector_f32(state, state.tensor_index.rope_freqs, "Gemma RoPE frequency factors");
  }
  const bool uses_swa_dimensions = state.model.attention_key_length_swa.has_value()
    && head_dim == state.model.attention_key_length_swa.value();
  if (uses_swa_dimensions) {
    rope_freq_factors.clear();
  }
  return GemmaLayerExecutionConfig {
    hidden_size,
    head_count,
    kv_head_count,
    head_dim,
    query_shape.row_count,
    key_shape.row_count,
    uses_swa_dimensions
      ? state.model.attention_sliding_window.value_or(0)
      : 0,
    uses_swa_dimensions
      ? state.model.rope_freq_base_swa.value_or(10000.0)
      : state.model.rope_freq_base.value_or(10000.0),
    state.model.rope_freq_scale.value_or(1.0),
    state.model.layer_norm_rms_epsilon.value_or(1e-5),
    rope_freq_factors,
    query_shape,
    key_shape,
    value_shape,
    output_shape,
    gate_shape,
    down_shape,
    up_shape,
    per_layer_input_size,
    per_layer_input_gate_shape,
    per_layer_projection_shape,
  };
}

static bool build_gemma_layer_execution_config_cache(LoadedModelState &state) {
  state.gemma_layer_configs.clear();
  state.gemma_layer_configs_ready = false;
  if (!is_gemma_architecture(state.model.architecture) || state.tensor_index.layers.empty()) {
    return false;
  }
  state.gemma_layer_configs.reserve(state.tensor_index.layers.size());
  for (const LlamaLayerTensorIndex &layer : state.tensor_index.layers) {
    const auto config = resolve_gemma_layer_execution_config(state, layer);
    if (!config.has_value()) {
      state.gemma_layer_configs.clear();
      return false;
    }
    state.gemma_layer_configs.push_back(config.value());
  }
  state.gemma_layer_configs_ready = state.gemma_layer_configs.size() == state.tensor_index.layers.size();
  return state.gemma_layer_configs_ready;
}

// Mirror of the Gemma layer-config cache for the plain llama decode path:
// resolve all seven projection shapes per layer once at load instead of per
// projection per token. Layers with unsupported/missing tensors leave the
// cache unready and the per-token path falls back to live resolution.
static bool build_llama_layer_shape_cache(LoadedModelState &state) {
  state.llama_layer_shapes.clear();
  state.llama_layer_shapes_ready = false;
  if (is_gemma_architecture(state.model.architecture) || state.tensor_index.layers.empty()) {
    return false;
  }
  state.llama_layer_shapes.reserve(state.tensor_index.layers.size());
  for (const LlamaLayerTensorIndex &layer : state.tensor_index.layers) {
    if (layer.query == k_missing_tensor_index
      || layer.key == k_missing_tensor_index
      || layer.value == k_missing_tensor_index
      || layer.output == k_missing_tensor_index
      || layer.gate == k_missing_tensor_index
      || layer.down == k_missing_tensor_index
      || layer.up == k_missing_tensor_index) {
      state.llama_layer_shapes.clear();
      return false;
    }
    try {
      LlamaLayerShapeConfig config;
      config.query_shape = tensor_matrix_shape(tensor_at(state, layer.query), "Llama attention query projection");
      config.key_shape = tensor_matrix_shape(tensor_at(state, layer.key), "Llama attention key projection");
      config.value_shape = tensor_matrix_shape(tensor_at(state, layer.value), "Llama attention value projection");
      config.output_shape = tensor_matrix_shape(tensor_at(state, layer.output), "Llama attention output projection");
      config.gate_shape = tensor_matrix_shape(tensor_at(state, layer.gate), "Llama MLP gate projection");
      config.down_shape = tensor_matrix_shape(tensor_at(state, layer.down), "Llama MLP down projection");
      config.up_shape = tensor_matrix_shape(tensor_at(state, layer.up), "Llama MLP up projection");
      state.llama_layer_shapes.push_back(config);
    } catch (const WorkerError &) {
      state.llama_layer_shapes.clear();
      return false;
    }
  }
  state.llama_layer_shapes_ready = state.llama_layer_shapes.size() == state.tensor_index.layers.size();
  return state.llama_layer_shapes_ready;
}

// Per-call cos/sin table for RoPE, keyed by everything the angle depends on.
// Both rope variants rotate (head_count x half_dim) elements but only half_dim
// distinct angles exist, so the table collapses the transcendental work by the
// head count. Cached across calls as well: prefill walks positions in order for
// 28 layers and Q then K, so consecutive calls frequently repeat a key.
struct RopeAngleTable {
  std::vector<float> cosines;
  std::vector<float> sines;
  std::uint64_t position = ~0ull;
  std::uint64_t head_dim = 0;
  double freq_base = 0.0;
  double freq_scale = 0.0;
  const void *factors_key = nullptr;
  std::size_t factors_size = 0;
};

static RopeAngleTable &rope_angle_table() {
  static thread_local RopeAngleTable table;
  return table;
}

static void rope_angle_table_for(
  std::uint64_t position,
  std::uint64_t head_dim,
  std::uint64_t half_dim,
  double rope_freq_base,
  double rope_freq_scale,
  const std::vector<float> &rope_freq_factors) {
  RopeAngleTable &table = rope_angle_table();
  if (table.position == position
    && table.head_dim == head_dim
    && table.freq_base == rope_freq_base
    && table.freq_scale == rope_freq_scale
    && table.factors_key == static_cast<const void *>(rope_freq_factors.data())
    && table.factors_size == rope_freq_factors.size()
    && table.cosines.size() == static_cast<std::size_t>(half_dim)) {
    return;
  }
  table.cosines.resize(static_cast<std::size_t>(half_dim));
  table.sines.resize(static_cast<std::size_t>(half_dim));
  for (std::uint64_t pair = 0; pair < half_dim; pair += 1) {
    const double exponent = static_cast<double>(pair * 2) / static_cast<double>(head_dim);
    double inv_freq = std::pow(rope_freq_base, -exponent);
    if (pair < rope_freq_factors.size()) {
      const float factor = rope_freq_factors[static_cast<std::size_t>(pair)];
      if (factor > 0.0f) {
        inv_freq /= static_cast<double>(factor);
      }
    }
    const double angle = static_cast<double>(position) * rope_freq_scale * inv_freq;
    table.cosines[static_cast<std::size_t>(pair)] = static_cast<float>(std::cos(angle));
    table.sines[static_cast<std::size_t>(pair)] = static_cast<float>(std::sin(angle));
  }
  table.position = position;
  table.head_dim = head_dim;
  table.freq_base = rope_freq_base;
  table.freq_scale = rope_freq_scale;
  table.factors_key = static_cast<const void *>(rope_freq_factors.data());
  table.factors_size = rope_freq_factors.size();
}

static void apply_rope_in_place(
  std::vector<float> &values,
  std::uint64_t position,
  std::uint64_t head_count,
  std::uint64_t head_dim,
  double rope_freq_base,
  double rope_freq_scale,
  const std::vector<float> &rope_freq_factors = {}) {
  if (values.size() != static_cast<std::size_t>(head_count * head_dim)) {
    throw WorkerError("NATIVE_WORKER_ERROR", "RoPE input shape does not match the configured attention dimensions.");
  }
  const std::uint64_t half_dim = head_dim / 2;
  // The rotation angle depends on (position, pair) only - not on head - so the
  // pow/cos/sin triple was being recomputed identically for every head. At 24
  // heads that is a 24x redundancy, and prefill calls this per token per layer
  // for both Q and K: ~15.4M triples per 269-token prefill, measured as 1.3-1.4 s
  // in prefill-rope-kv-append. Computing the table once per call and reusing it
  // across heads produces bit-identical results.
  rope_angle_table_for(
    position, head_dim, half_dim, rope_freq_base, rope_freq_scale, rope_freq_factors);
  const RopeAngleTable &table = rope_angle_table();
  for (std::uint64_t head = 0; head < head_count; head += 1) {
    const std::size_t head_offset = static_cast<std::size_t>(head * head_dim);
    for (std::uint64_t pair = 0; pair < half_dim; pair += 1) {
      const std::size_t even_index = head_offset + static_cast<std::size_t>(pair * 2);
      const std::size_t odd_index = even_index + 1;
      const float cosine = table.cosines[static_cast<std::size_t>(pair)];
      const float sine = table.sines[static_cast<std::size_t>(pair)];
      const float even = values[even_index];
      const float odd = values[odd_index];
      values[even_index] = even * cosine - odd * sine;
      values[odd_index] = even * sine + odd * cosine;
    }
  }
}

static void apply_neox_rope_in_place(
  std::vector<float> &values,
  std::uint64_t position,
  std::uint64_t head_count,
  std::uint64_t head_dim,
  double rope_freq_base,
  double rope_freq_scale,
  const std::vector<float> &rope_freq_factors = {}) {
  if (values.size() != static_cast<std::size_t>(head_count * head_dim) || head_dim % 2 != 0) {
    throw WorkerError("NATIVE_WORKER_ERROR", "NEOX RoPE input shape does not match the configured attention dimensions.");
  }
  const std::uint64_t half_dim = head_dim / 2;
  // Same hoist as the interleaved variant above: the angle is head-independent.
  rope_angle_table_for(
    position, head_dim, half_dim, rope_freq_base, rope_freq_scale, rope_freq_factors);
  const RopeAngleTable &table = rope_angle_table();
  for (std::uint64_t head = 0; head < head_count; head += 1) {
    const std::size_t head_offset = static_cast<std::size_t>(head * head_dim);
    for (std::uint64_t pair = 0; pair < half_dim; pair += 1) {
      const std::size_t first_index = head_offset + static_cast<std::size_t>(pair);
      const std::size_t second_index = first_index + static_cast<std::size_t>(half_dim);
      const float cosine = table.cosines[static_cast<std::size_t>(pair)];
      const float sine = table.sines[static_cast<std::size_t>(pair)];
      const float first = values[first_index];
      const float second = values[second_index];
      values[first_index] = first * cosine - second * sine;
      values[second_index] = first * sine + second * cosine;
    }
  }
}

// RoPE for the family path: supports both rotation layouts, partial rotary
// (only the first rope_dim dims of each head rotate, e.g. phi/glm4), optional
// frequency factors (llama-3 rope_freqs or phi3 long/short factors), and the
// longrope attention magnitude factor.
static void apply_family_rope_in_place(
  std::vector<float> &values,
  std::uint64_t position,
  std::uint64_t head_count,
  std::uint64_t head_dim,
  std::uint64_t rope_dim,
  FamilyRopeKind rope_kind,
  double rope_freq_base,
  double rope_freq_scale,
  const std::vector<float> &rope_freq_factors,
  double attn_factor) {
  if (values.size() != static_cast<std::size_t>(head_count * head_dim)
    || rope_dim == 0
    || rope_dim > head_dim
    || rope_dim % 2 != 0) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Family RoPE input shape does not match the configured attention dimensions.");
  }
  const std::uint64_t half_dim = rope_dim / 2;
  const float magnitude = static_cast<float>(attn_factor > 0.0 ? attn_factor : 1.0);
  for (std::uint64_t head = 0; head < head_count; head += 1) {
    const std::size_t head_offset = static_cast<std::size_t>(head * head_dim);
    for (std::uint64_t pair = 0; pair < half_dim; pair += 1) {
      const std::size_t first_index = rope_kind == FamilyRopeKind::neox
        ? head_offset + static_cast<std::size_t>(pair)
        : head_offset + static_cast<std::size_t>(pair * 2);
      const std::size_t second_index = rope_kind == FamilyRopeKind::neox
        ? first_index + static_cast<std::size_t>(half_dim)
        : first_index + 1;
      const double exponent = static_cast<double>(pair * 2) / static_cast<double>(rope_dim);
      double inv_freq = std::pow(rope_freq_base, -exponent);
      if (pair < rope_freq_factors.size()) {
        const float factor = rope_freq_factors[static_cast<std::size_t>(pair)];
        if (factor > 0.0f) {
          inv_freq /= static_cast<double>(factor);
        }
      }
      const double angle = static_cast<double>(position) * rope_freq_scale * inv_freq;
      const float cosine = static_cast<float>(std::cos(angle)) * magnitude;
      const float sine = static_cast<float>(std::sin(angle)) * magnitude;
      const float first = values[first_index];
      const float second = values[second_index];
      values[first_index] = first * cosine - second * sine;
      values[second_index] = first * sine + second * cosine;
    }
  }
}

static void scale_vector_in_place(std::vector<float> &values, double scale) {
  const float factor = static_cast<float>(scale);
  for (float &value : values) {
    value *= factor;
  }
}

static void apply_logit_softcap_in_place(std::vector<float> &logits, double softcap) {
  if (softcap <= 0.0) {
    return;
  }
  const float cap = static_cast<float>(softcap);
  for (float &logit : logits) {
    logit = cap * std::tanh(logit / cap);
  }
}

static std::vector<float> build_gemma_per_layer_inputs(
  const LoadedModelState &state,
  std::uint64_t token_id,
  const std::vector<float> &scaled_token_embedding) {
  const std::uint64_t per_layer_input_size = state.model.embedding_length_per_layer_input.value_or(0);
  if (per_layer_input_size == 0) {
    return {};
  }
  const std::uint64_t layer_count = static_cast<std::uint64_t>(state.tensor_index.layers.size());
  const std::uint64_t expected_size = per_layer_input_size * layer_count;
  const TensorInfo &token_embeddings = tensor_at(state, state.tensor_index.per_layer_token_embeddings);
  const TensorInfo &model_projection = tensor_at(state, state.tensor_index.per_layer_model_projection);
  const MatrixShape projection_shape = tensor_matrix_shape(model_projection, "Gemma packed per-layer model projection");
  if (projection_shape.row_width != scaled_token_embedding.size()
    || projection_shape.row_count != expected_size
    || token_embeddings.dimensions.size() < 2
    || token_embeddings.dimensions[0] != expected_size) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Gemma packed per-layer input tensors have incompatible shapes.");
  }

  std::vector<float> token_inputs = tensor_row_f32(state, token_embeddings, token_id);
  std::vector<float> projected_inputs = matrix_vector_mul_tensor(
    state,
    model_projection,
    projection_shape,
    scaled_token_embedding,
    "Gemma packed per-layer model projection");
  const std::vector<float> &projection_norm = cached_tensor_vector_f32(
    state,
    state.tensor_index.per_layer_projection_norm,
    "Gemma per-layer projection norm");
  if (projection_norm.size() != static_cast<std::size_t>(per_layer_input_size)) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Gemma per-layer projection norm has an incompatible shape.");
  }

  scale_vector_in_place(token_inputs, std::sqrt(static_cast<double>(per_layer_input_size)));
  scale_vector_in_place(projected_inputs, 1.0 / std::sqrt(static_cast<double>(scaled_token_embedding.size())));
  std::vector<float> combined(static_cast<std::size_t>(expected_size), 0.0f);
  constexpr float input_scale = 0.7071067811865475f;
  const double epsilon = state.model.layer_norm_rms_epsilon.value_or(1e-5);
  for (std::uint64_t layer = 0; layer < layer_count; layer += 1) {
    const std::size_t offset = static_cast<std::size_t>(layer * per_layer_input_size);
    double sum = 0.0;
    for (std::uint64_t item = 0; item < per_layer_input_size; item += 1) {
      const float value = projected_inputs[offset + static_cast<std::size_t>(item)];
      sum += static_cast<double>(value) * static_cast<double>(value);
    }
    const double mean = sum / static_cast<double>(per_layer_input_size);
    const float norm_scale = static_cast<float>(1.0 / std::sqrt(mean + epsilon));
    for (std::uint64_t item = 0; item < per_layer_input_size; item += 1) {
      const std::size_t index = offset + static_cast<std::size_t>(item);
      const float projected = projected_inputs[index] * norm_scale * projection_norm[static_cast<std::size_t>(item)];
      combined[index] = (projected + token_inputs[index]) * input_scale;
    }
  }
  return combined;
}

static FloatBatch build_gemma_per_layer_inputs_batch(
  const LoadedModelState &state,
  const std::vector<std::uint64_t> &token_ids,
  const FloatBatch &scaled_token_embeddings,
  BatchExecutionConfig *execution_config) {
  const std::uint64_t per_layer_input_size = state.model.embedding_length_per_layer_input.value_or(0);
  if (per_layer_input_size == 0) {
    return FloatBatch(token_ids.size());
  }
  if (token_ids.size() != scaled_token_embeddings.size()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Gemma per-layer input batch size is inconsistent.");
  }
  const std::uint64_t layer_count = static_cast<std::uint64_t>(state.tensor_index.layers.size());
  const std::uint64_t expected_size = per_layer_input_size * layer_count;
  const TensorInfo &token_embeddings = tensor_at(state, state.tensor_index.per_layer_token_embeddings);
  const TensorInfo &model_projection = tensor_at(state, state.tensor_index.per_layer_model_projection);
  const MatrixShape projection_shape = tensor_matrix_shape(
    model_projection,
    "Gemma packed per-layer model projection");
  if (scaled_token_embeddings.empty()
    || projection_shape.row_width != scaled_token_embeddings.front().size()
    || projection_shape.row_count != expected_size
    || token_embeddings.dimensions.size() < 2
    || token_embeddings.dimensions[0] != expected_size) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Gemma packed per-layer input batch tensors have incompatible shapes.");
  }

  FloatBatch token_inputs;
  token_inputs.reserve(token_ids.size());
  for (std::uint64_t token_id : token_ids) {
    token_inputs.push_back(tensor_row_f32(state, token_embeddings, token_id));
  }
  FloatBatch projected_inputs = matrix_batch_mul_tensor(
    state,
    model_projection,
    projection_shape,
    scaled_token_embeddings,
    execution_config,
    "Gemma packed per-layer model projection batch");
  const std::vector<float> &projection_norm = cached_tensor_vector_f32(
    state,
    state.tensor_index.per_layer_projection_norm,
    "Gemma per-layer projection norm");
  if (projection_norm.size() != static_cast<std::size_t>(per_layer_input_size)) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Gemma per-layer projection norm has an incompatible shape.");
  }

  constexpr float input_scale = 0.7071067811865475f;
  const double epsilon = state.model.layer_norm_rms_epsilon.value_or(1e-5);
  for (std::size_t batch_index = 0; batch_index < token_ids.size(); batch_index += 1) {
    scale_vector_in_place(token_inputs[batch_index], std::sqrt(static_cast<double>(per_layer_input_size)));
    scale_vector_in_place(
      projected_inputs[batch_index],
      1.0 / std::sqrt(static_cast<double>(scaled_token_embeddings[batch_index].size())));
    for (std::uint64_t layer = 0; layer < layer_count; layer += 1) {
      const std::size_t offset = static_cast<std::size_t>(layer * per_layer_input_size);
      double sum = 0.0;
      for (std::uint64_t item = 0; item < per_layer_input_size; item += 1) {
        const float value = projected_inputs[batch_index][offset + static_cast<std::size_t>(item)];
        sum += static_cast<double>(value) * static_cast<double>(value);
      }
      const float norm_scale = static_cast<float>(
        1.0 / std::sqrt(sum / static_cast<double>(per_layer_input_size) + epsilon));
      for (std::uint64_t item = 0; item < per_layer_input_size; item += 1) {
        const std::size_t index = offset + static_cast<std::size_t>(item);
        const float projected = projected_inputs[batch_index][index]
          * norm_scale
          * projection_norm[static_cast<std::size_t>(item)];
        projected_inputs[batch_index][index] =
          (projected + token_inputs[batch_index][index]) * input_scale;
      }
    }
  }
  return projected_inputs;
}

static void softmax_in_place(std::vector<float> &weights) {
  if (weights.empty()) {
    return;
  }
  const float max_value = *std::max_element(weights.begin(), weights.end());
  double sum = 0.0;
  for (std::size_t index = 0; index < weights.size(); index += 1) {
    const double weight = std::exp(static_cast<double>(weights[index] - max_value));
    weights[index] = static_cast<float>(weight);
    sum += weight;
  }
  if (sum == 0.0) {
    return;
  }
  for (float &weight : weights) {
    weight = static_cast<float>(static_cast<double>(weight) / sum);
  }
}

static std::vector<float> softmax(const std::vector<float> &input) {
  std::vector<float> weights = input;
  softmax_in_place(weights);
  return weights;
}

static SessionState::SamplingState parse_sampling_state(const JsonObject &params) {
  SessionState::SamplingState sampling;
  const JsonObject *sampling_params = find_object_field(params, "sampling");
  if (sampling_params != nullptr) {
    sampling.temperature = json_number_or_default(*sampling_params, "temperature", 0.0);
    sampling.top_k = static_cast<std::uint64_t>(json_number_or_default(*sampling_params, "topK", 0));
    sampling.top_p = json_number_or_default(*sampling_params, "topP", 1.0);
    sampling.seed = static_cast<std::uint64_t>(json_number_or_default(*sampling_params, "seed", 0));
  }
  if (!std::isfinite(sampling.temperature) || sampling.temperature < 0.0) {
    sampling.temperature = 0.0;
  }
  if (!std::isfinite(sampling.top_p) || sampling.top_p <= 0.0 || sampling.top_p > 1.0) {
    sampling.top_p = 1.0;
  }
  sampling.rng.seed(sampling.seed);
  sampling.initialized = true;
  return sampling;
}

static std::uint64_t select_next_token(const std::vector<float> &logits, SessionState::SamplingState &sampling) {
  // Also part of the unaccounted 53 ms/token. A greedy pass is one sweep of a
  // 128256-entry vector; the temperature path additionally sorts a candidate
  // slice, and a full indirect sort of this vocab used to cost ~30 ms per token
  // before it was capped (see the note in the sampling helper below).
  const DecodePhaseScope phase_scope("sampling");
  if (logits.empty()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Token selection requires non-empty logits.");
  }
  if (sampling.temperature <= 0.0) {
    std::size_t best = 0;
    for (std::size_t index = 1; index < logits.size(); index += 1) {
      if (logits[index] > logits[best]) {
        best = index;
      }
    }
    return static_cast<std::uint64_t>(best);
  }

  std::vector<std::size_t> indices(logits.size());
  for (std::size_t index = 0; index < logits.size(); index += 1) {
    indices[index] = index;
  }
  // A full indirect sort of the vocab (262k for gemma) costs ~30 ms per
  // sampled token - measured as the gap between GPU graph time and
  // decodeMs whenever temperature > 0. Only the top slice can ever be
  // sampled: partial_sort exactly reproduces the top_k behavior, and when
  // top_k is unset the sort is capped at 1024 candidates - after
  // temperature scaling and softmax the probability mass beyond the top
  // 1024 of a 262k vocab is far below one float ulp, so the sampled
  // distribution is unchanged in practice.
  const std::size_t sort_limit = sampling.top_k > 0
    ? std::min<std::size_t>(static_cast<std::size_t>(sampling.top_k), indices.size())
    : std::min<std::size_t>(indices.size(), 1024);
  if (sort_limit < indices.size()) {
    std::partial_sort(
      indices.begin(),
      indices.begin() + static_cast<std::ptrdiff_t>(sort_limit),
      indices.end(),
      [&](std::size_t left, std::size_t right) {
        return logits[left] > logits[right];
      });
    indices.resize(sort_limit);
  } else {
    std::sort(indices.begin(), indices.end(), [&](std::size_t left, std::size_t right) {
      return logits[left] > logits[right];
    });
  }

  if (sampling.top_k > 0 && sampling.top_k < indices.size()) {
    indices.resize(static_cast<std::size_t>(sampling.top_k));
  }

  std::vector<float> scaled_logits;
  scaled_logits.reserve(indices.size());
  for (std::size_t index : indices) {
    scaled_logits.push_back(static_cast<float>(static_cast<double>(logits[index]) / sampling.temperature));
  }
  std::vector<float> probabilities = softmax(scaled_logits);

  if (sampling.top_p < 1.0 && !probabilities.empty()) {
    double cumulative = 0.0;
    std::size_t keep_count = 0;
    for (; keep_count < probabilities.size(); keep_count += 1) {
      cumulative += probabilities[keep_count];
      if (cumulative >= sampling.top_p) {
        keep_count += 1;
        break;
      }
    }
    keep_count = std::max<std::size_t>(1, std::min<std::size_t>(keep_count, probabilities.size()));
    indices.resize(keep_count);
    probabilities.resize(keep_count);
    double kept_sum = 0.0;
    for (float probability : probabilities) {
      kept_sum += probability;
    }
    if (kept_sum > 0.0) {
      for (float &probability : probabilities) {
        probability = static_cast<float>(static_cast<double>(probability) / kept_sum);
      }
    }
  }

  std::vector<double> weights;
  weights.reserve(probabilities.size());
  for (float probability : probabilities) {
    weights.push_back(static_cast<double>(probability));
  }
  std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
  const std::size_t sampled_offset = distribution(sampling.rng);
  return static_cast<std::uint64_t>(indices[sampled_offset]);
}

static float silu(float value) {
  return value / (1.0f + std::exp(-value));
}

static std::vector<float> add_vectors(const std::vector<float> &left, const std::vector<float> &right, const std::string &label) {
  if (left.size() != right.size()) {
    throw WorkerError("NATIVE_WORKER_ERROR", label + " requires vectors with the same size.");
  }
  std::vector<float> output(left.size(), 0.0f);
  for (std::size_t index = 0; index < left.size(); index += 1) {
    output[index] = left[index] + right[index];
  }
  return output;
}

static std::vector<float> project_hidden_to_logits(
  const LoadedModelState &state,
  const TensorInfo &output_tensor,
  const std::vector<float> &hidden_state) {
  // Part of the 53 ms/token that no phase covered. This is a 128256 x 3072 Q4_K
  // matvec - about 14 MB, a third of a whole layer's weight traffic - and it runs
  // once per token outside the layer loop. At the measured 20.2 GB/s plus 0.84 ms
  // of submission cost it should land near 1.5 ms; anything much above that is
  // upstream of the GPU.
  const auto lm_head_start = std::chrono::steady_clock::now();
  const MatrixShape shape = tensor_matrix_shape(output_tensor, "LM head projection");
  std::vector<float> logits = matrix_vector_mul_tensor(
    state, output_tensor, shape, hidden_state, "LM head projection");
  record_decode_phase_since("lm-head", lm_head_start);
  return logits;
}

static std::size_t argmax_index(const std::vector<float> &values) {
  if (values.empty()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "argmax requires a non-empty vector.");
  }
  std::size_t best = 0;
  for (std::size_t index = 1; index < values.size(); index += 1) {
    if (values[index] > values[best]) {
      best = index;
    }
  }
  return best;
}

static bool layered_llama_graph_executable(const LoadedModelState &state) {
  if (state.model.block_count.value_or(0) == 0) {
    return false;
  }
  // Family-spec models (qwen2 biases, MoE routing, ...) must never run
  // through the plain llama graph: it would execute but produce silently
  // wrong output. They route through the family path instead.
  if (resolve_family_spec(state.model).has_value()) {
    return false;
  }
  if (!state.tensor_index.ready || state.model.token_vocabulary.empty()) {
    return false;
  }
  const auto config = resolve_llama_execution_config(state);
  if (!config.has_value()) {
    return false;
  }
  const auto tensor_index_supported = [&](std::size_t index) {
    return index != k_missing_tensor_index && tensor_type_supported_for_execution(tensor_at(state, index));
  };
  if (!tensor_index_supported(state.tensor_index.token_embeddings)
    || !tensor_index_supported(state.tensor_index.output_norm)
    || !tensor_index_supported(state.tensor_index.output)) {
    return false;
  }
  for (const LlamaLayerTensorIndex &layer : state.tensor_index.layers) {
    if (!tensor_index_supported(layer.attention_norm)
      || !tensor_index_supported(layer.query)
      || !tensor_index_supported(layer.key)
      || !tensor_index_supported(layer.value)
      || !tensor_index_supported(layer.output)
      || !tensor_index_supported(layer.ffn_norm)
      || !tensor_index_supported(layer.gate)
      || !tensor_index_supported(layer.down)
      || !tensor_index_supported(layer.up)) {
      return false;
    }
    try {
      const MatrixShape query_shape = tensor_matrix_shape(tensor_at(state, layer.query), "Llama attention query projection");
      const MatrixShape key_shape = tensor_matrix_shape(tensor_at(state, layer.key), "Llama attention key projection");
      const MatrixShape value_shape = tensor_matrix_shape(tensor_at(state, layer.value), "Llama attention value projection");
      const MatrixShape output_shape = tensor_matrix_shape(tensor_at(state, layer.output), "Llama attention output projection");
      const MatrixShape gate_shape = tensor_matrix_shape(tensor_at(state, layer.gate), "Llama MLP gate projection");
      const MatrixShape down_shape = tensor_matrix_shape(tensor_at(state, layer.down), "Llama MLP down projection");
      const MatrixShape up_shape = tensor_matrix_shape(tensor_at(state, layer.up), "Llama MLP up projection");
      const std::uint64_t kv_hidden_size = config->kv_head_count * config->head_dim;
      if (query_shape.row_width != config->hidden_size || query_shape.row_count != config->hidden_size) return false;
      if (key_shape.row_width != config->hidden_size || key_shape.row_count != kv_hidden_size) return false;
      if (value_shape.row_width != config->hidden_size || value_shape.row_count != kv_hidden_size) return false;
      if (output_shape.row_width != config->hidden_size || output_shape.row_count != config->hidden_size) return false;
      if (gate_shape.row_width != config->hidden_size || up_shape.row_width != config->hidden_size || gate_shape.row_count != up_shape.row_count) return false;
      if (down_shape.row_width != gate_shape.row_count || down_shape.row_count != config->hidden_size) return false;
    } catch (const std::exception &) {
      return false;
    }
  }
  return true;
}

static bool layered_gemma_graph_executable(const LoadedModelState &state) {
  if (!is_gemma_architecture(state.model.architecture) || state.model.block_count.value_or(0) == 0) {
    return false;
  }
  if (!state.tensor_index.ready || state.model.token_vocabulary.empty()) {
    return false;
  }
  const auto tensor_index_supported = [&](std::size_t index) {
    return index != k_missing_tensor_index && tensor_type_supported_for_execution(tensor_at(state, index));
  };
  if (!tensor_index_supported(state.tensor_index.token_embeddings)
    || !tensor_index_supported(state.tensor_index.output_norm)
    || !tensor_index_supported(state.tensor_index.output)) {
    return false;
  }
  const std::uint64_t per_layer_input_size = state.model.embedding_length_per_layer_input.value_or(0);
  if (per_layer_input_size > 0) {
    if (!tensor_index_supported(state.tensor_index.per_layer_token_embeddings)
      || !tensor_index_supported(state.tensor_index.per_layer_model_projection)
      || !tensor_index_supported(state.tensor_index.per_layer_projection_norm)) {
      return false;
    }
    try {
      const std::uint64_t expected_packed_size = per_layer_input_size
        * static_cast<std::uint64_t>(state.tensor_index.layers.size());
      const TensorInfo &per_layer_token_embeddings = tensor_at(state, state.tensor_index.per_layer_token_embeddings);
      const TensorInfo &per_layer_projection_norm = tensor_at(state, state.tensor_index.per_layer_projection_norm);
      const MatrixShape per_layer_model_projection = tensor_matrix_shape(
        tensor_at(state, state.tensor_index.per_layer_model_projection),
        "Gemma packed per-layer model projection");
      if (per_layer_token_embeddings.dimensions.size() < 2
        || per_layer_token_embeddings.dimensions[0] != expected_packed_size
        || per_layer_model_projection.row_width != state.model.embedding_length.value_or(0)
        || per_layer_model_projection.row_count != expected_packed_size
        || per_layer_projection_norm.dimensions.size() != 1
        || per_layer_projection_norm.element_count != per_layer_input_size) {
        return false;
      }
    } catch (const std::exception &) {
      return false;
    }
  }
  for (std::size_t layer_number = 0; layer_number < state.tensor_index.layers.size(); layer_number += 1) {
    const LlamaLayerTensorIndex &layer = state.tensor_index.layers[layer_number];
    if (!tensor_index_supported(layer.attention_norm)
      || !tensor_index_supported(layer.query)
      || !tensor_index_supported(layer.key)
      || !tensor_index_supported(layer.value)
      || !tensor_index_supported(layer.output)
      || !tensor_index_supported(layer.query_norm)
      || !tensor_index_supported(layer.key_norm)
      || !tensor_index_supported(layer.ffn_norm)
      || !tensor_index_supported(layer.gate)
      || !tensor_index_supported(layer.down)
      || !tensor_index_supported(layer.up)
      || !tensor_index_supported(layer.post_attention_norm)
      || !tensor_index_supported(layer.post_ffw_norm)
      || (per_layer_input_size > 0 && !tensor_index_supported(layer.per_layer_input_gate))
      || (per_layer_input_size > 0 && !tensor_index_supported(layer.per_layer_projection))
      || (per_layer_input_size > 0 && !tensor_index_supported(layer.post_norm))
      || (per_layer_input_size > 0 && !tensor_index_supported(layer.layer_output_scale))) {
      return false;
    }
    try {
      std::optional<GemmaLayerExecutionConfig> resolved_config;
      const GemmaLayerExecutionConfig *config = nullptr;
      if (state.gemma_layer_configs_ready && layer_number < state.gemma_layer_configs.size()) {
        config = &state.gemma_layer_configs[layer_number];
      } else {
        resolved_config = resolve_gemma_layer_execution_config(state, layer);
        if (!resolved_config.has_value()) return false;
        config = &resolved_config.value();
      }
      const TensorInfo &post_attention_norm_tensor = tensor_at(state, layer.post_attention_norm);
      const TensorInfo &post_ffw_norm_tensor = tensor_at(state, layer.post_ffw_norm);
      if (config->gate_shape.row_width != config->hidden_size
        || config->up_shape.row_width != config->hidden_size
        || config->gate_shape.row_count != config->up_shape.row_count
        || config->down_shape.row_width != config->gate_shape.row_count
        || config->down_shape.row_count != config->hidden_size
        || post_attention_norm_tensor.dimensions.size() != 1
        || post_attention_norm_tensor.element_count != config->hidden_size
        || post_ffw_norm_tensor.dimensions.size() != 1
        || post_ffw_norm_tensor.element_count != config->hidden_size) {
        return false;
      }
    } catch (const std::exception &) {
      return false;
    }
  }
  return true;
}

static bool zero_layer_graph_executable(const LoadedModelState &state) {
  if (state.model.block_count.value_or(0) != 0) {
    return false;
  }
  if (state.tensor_index.token_embeddings == k_missing_tensor_index
    || state.tensor_index.output_norm == k_missing_tensor_index
    || state.tensor_index.output == k_missing_tensor_index) {
    return false;
  }
  if (state.model.token_vocabulary.empty()) {
    return false;
  }
  const TensorInfo &token_embeddings = tensor_at(state, state.tensor_index.token_embeddings);
  const TensorInfo &output_norm = tensor_at(state, state.tensor_index.output_norm);
  const TensorInfo &output = tensor_at(state, state.tensor_index.output);
  return tensor_type_supported_for_execution(token_embeddings)
    && tensor_type_supported_for_execution(output_norm)
    && tensor_type_supported_for_execution(output);
}

static bool llama_graph_executable(const LoadedModelState &state) {
  return zero_layer_graph_executable(state) || layered_llama_graph_executable(state);
}

// Readiness gate for the family expansion path. Requires a resolvable spec
// and execution config, a complete family tensor directory (index.ready uses
// the family requirement list for spec models), tokenizer vocabulary, and
// supported tensor types on every projection the forward pass will touch.
static bool layered_family_graph_executable(const LoadedModelState &state) {
  const auto spec = resolve_family_spec(state.model);
  if (!spec.has_value() || state.model.block_count.value_or(0) == 0) {
    return false;
  }
  if (!state.tensor_index.ready || state.model.token_vocabulary.empty()) {
    return false;
  }
  if (!resolve_family_execution_config(state).has_value()) {
    return false;
  }
  const auto tensor_index_supported = [&](std::size_t index) {
    return index != k_missing_tensor_index && tensor_type_supported_for_execution(tensor_at(state, index));
  };
  const auto optional_tensor_supported = [&](std::size_t index) {
    return index == k_missing_tensor_index || tensor_type_supported_for_execution(tensor_at(state, index));
  };
  if (!tensor_index_supported(state.tensor_index.token_embeddings)
    || !tensor_index_supported(state.tensor_index.output_norm)
    || !tensor_index_supported(state.tensor_index.output)) {
    return false;
  }
  for (const LlamaLayerTensorIndex &layer : state.tensor_index.layers) {
    if (!tensor_index_supported(layer.attention_norm) || !tensor_index_supported(layer.output)) {
      return false;
    }
    const bool fused_qkv_supported = tensor_index_supported(layer.qkv_fused);
    const bool split_qkv_supported = tensor_index_supported(layer.query)
      && tensor_index_supported(layer.key)
      && tensor_index_supported(layer.value);
    if (!fused_qkv_supported && !split_qkv_supported) {
      return false;
    }
    if (spec->qk_norm && (!tensor_index_supported(layer.query_norm) || !tensor_index_supported(layer.key_norm))) {
      return false;
    }
    if (spec->post_attention_norm && !tensor_index_supported(layer.post_attention_norm)) {
      return false;
    }
    if (spec->post_ffw_norm && !tensor_index_supported(layer.post_ffw_norm)) {
      return false;
    }
    if (spec->has_ffn_norm && !tensor_index_supported(layer.ffn_norm)) {
      return false;
    }
    const bool moe_layer = spec->moe && layer.moe_gate_exps != k_missing_tensor_index;
    if (moe_layer) {
      if (!tensor_index_supported(layer.moe_gate_inp)
        || !tensor_index_supported(layer.moe_gate_exps)
        || !tensor_index_supported(layer.moe_up_exps)
        || !tensor_index_supported(layer.moe_down_exps)
        || !optional_tensor_supported(layer.shexp_gate)
        || !optional_tensor_supported(layer.shexp_up)
        || !optional_tensor_supported(layer.shexp_down)) {
        return false;
      }
    } else {
      if (!tensor_index_supported(layer.down) || !tensor_index_supported(layer.up)) {
        return false;
      }
      if (spec->ffn_kind == FamilyFfnKind::swiglu_split && !tensor_index_supported(layer.gate)) {
        return false;
      }
    }
  }
  return true;
}

static bool native_graph_executable(const LoadedModelState &state) {
  return llama_graph_executable(state)
    || layered_gemma_graph_executable(state)
    || layered_family_graph_executable(state);
}

static std::string native_model_status(const LoadedModelState *state) {
  if (state == nullptr) {
    return "idle";
  }
  if (zero_layer_graph_executable(*state)) {
    return "minimal-graph-ready";
  }
  if (layered_llama_graph_executable(*state)) {
    return "llama-graph-ready";
  }
  if (layered_gemma_graph_executable(*state)) {
    return "gemma-graph-ready";
  }
  if (layered_family_graph_executable(*state)) {
    const auto spec = resolve_family_spec(state->model);
    return (spec.has_value() ? spec->family : "family") + "-graph-ready";
  }
  return "load-only";
}

// Globally unique GPU KV cache generation ids. A single counter (instead of
// per-call-site statics) guarantees two different session states can never
// alias the same D3D12 KV cache_id.
static std::uint64_t next_gpu_kv_cache_generation_value() {
  static std::atomic<std::uint64_t> next_generation {1};
  return next_generation.fetch_add(1, std::memory_order_relaxed);
}

static void reset_layered_session_state(SessionState &session, std::size_t layer_count) {
  session.sequence_token_ids.clear();
  session.layered_kv_cache.assign(layer_count, SessionState::LayerKvCache {});
  session.gpu_kv_cache_generation = next_gpu_kv_cache_generation_value();
  session.gpu_kv_max_positions = 0;
  session.gpu_kv_cache_active.assign(layer_count, false);
  // layered_kv_cache was just reassigned, so the cached graph descriptors'
  // initial_keys/initial_values pointers are stale.
  session.gpu_token_graph_layers.clear();
  session.gpu_token_graph_descriptors_ready = false;
  session.last_hidden_state.clear();
  session.last_logits.clear();
  session.cached_token_count = 0;
  session.generated_token_count = 0;
  session.last_prefill_ms = 0;
  session.last_decode_ms = 0;
  session.prefill_complete = false;
  session.decode_qkv_fuse_blocked = false;
  session.decode_qkv_fuse_detail.clear();
  session.decode_ffn_fuse_blocked = false;
  session.decode_ffn_fuse_detail.clear();
  session.decode_layer_tail_fuse_blocked = false;
  session.decode_layer_tail_fuse_detail.clear();
  session.sampling = SessionState::SamplingState {};
}

static void append_layer_cache(
  SessionState::LayerKvCache &cache,
  const std::vector<float> &key,
  const std::vector<float> &value) {
  cache.keys.append(key.data(), key.size());
  cache.values.append(value.data(), value.size());
}

// --- Prefix KV reuse (enhancement Phase 2.3) --------------------------------
// A new prefill whose token ids share a prefix with the session's cached
// sequence can keep that prefix's KV entries and only replay the divergent
// suffix. Multi-turn chats resend the whole growing transcript every turn, so
// this converts each turn's prefill cost from O(transcript) to O(new turn).
// RoPE is applied with absolute positions before keys enter the cache and
// both batched prefill paths already continue from cached_token_count, so a
// kept prefix stays valid without recomputation. LIZARD_NATIVE_PREFIX_REUSE=0
// disables the optimization (every prefill resets and replays, the historical
// behaviour).
static bool kv_prefix_reuse_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_PREFIX_REUSE");
    return value == nullptr || std::string(value) != "0";
  }();
  return enabled;
}

// Attempts to keep the longest common prefix of `previous_sequence` and
// `prompt_token_ids` in the session KV cache. Returns the number of prompt
// tokens whose KV entries were kept (0 = nothing reused; caller must run the
// full reset + replay). On success the session bookkeeping is rewound to the
// kept prefix and every GPU-side cache is invalidated, so the caller only
// consumes prompt tokens from the returned index onward.
static std::uint64_t try_reuse_kv_prefix(
  SessionState &session,
  const std::vector<std::uint64_t> &previous_sequence,
  std::uint64_t previous_cached_count,
  const std::vector<std::uint64_t> &prompt_token_ids,
  std::size_t layer_count) {
  if (!kv_prefix_reuse_enabled()
    || previous_cached_count == 0
    || previous_sequence.size() != static_cast<std::size_t>(previous_cached_count)
    || session.layered_kv_cache.size() != layer_count
    || prompt_token_ids.empty()) {
    return 0;
  }

  std::size_t common = 0;
  const std::size_t max_common = std::min(previous_sequence.size(), prompt_token_ids.size());
  while (common < max_common && previous_sequence[common] == prompt_token_ids[common]) {
    common += 1;
  }
  if (common == prompt_token_ids.size()) {
    // Identical (or fully-contained) prompt: re-consume the final token so
    // last_hidden_state/last_logits are freshly populated for decode.
    common -= 1;
  }
  if (common == 0) {
    return 0;
  }

  // Validate every lane divides evenly into per-position rows before mutating
  // anything, so a half-truncated cache can never be left behind.
  for (const SessionState::LayerKvCache &layer : session.layered_kv_cache) {
    const std::size_t key_size = layer.keys.size();
    const std::size_t value_size = layer.values.size();
    if (key_size != value_size) {
      return 0;
    }
    if (key_size != 0 && key_size % static_cast<std::size_t>(previous_cached_count) != 0) {
      return 0;
    }
  }

  for (SessionState::LayerKvCache &layer : session.layered_kv_cache) {
    const std::size_t lane_size = layer.keys.size();
    if (lane_size == 0) {
      continue;
    }
    const std::size_t stride = lane_size / static_cast<std::size_t>(previous_cached_count);
    layer.keys.truncate(common * stride);
    layer.values.truncate(common * stride);
  }

  session.sequence_token_ids.assign(
    prompt_token_ids.begin(),
    prompt_token_ids.begin() + static_cast<std::ptrdiff_t>(common));
  session.cached_token_count = common;
  session.generated_token_count = 0;
  session.last_hidden_state.clear();
  session.last_logits.clear();
  session.prefill_complete = false;
  session.last_prefill_ms = 0;
  session.last_decode_ms = 0;
  session.sampling = SessionState::SamplingState {};
  session.decode_qkv_fuse_blocked = false;
  session.decode_qkv_fuse_detail.clear();
  session.decode_ffn_fuse_blocked = false;
  session.decode_ffn_fuse_detail.clear();
  session.decode_layer_tail_fuse_blocked = false;
  session.decode_layer_tail_fuse_detail.clear();

  // GPU KV caches were seeded from the old sequence: force a fresh generation
  // so the resident lanes re-seed from the truncated CPU lanes on next use.
  session.gpu_kv_cache_generation = next_gpu_kv_cache_generation_value();
  session.gpu_kv_max_positions = 0;
  session.gpu_kv_cache_active.assign(layer_count, false);
  session.gpu_token_graph_layers.clear();
  session.gpu_token_graph_descriptors_ready = false;

  return static_cast<std::uint64_t>(common);
}

// Rewinds the session KV cache and token bookkeeping to the first
// `keep_positions` positions. Used by speculative decoding to discard the KV
// entries of rejected draft tokens. GPU-side caches are invalidated the same
// way prefix reuse does it.
static void truncate_session_kv_positions(SessionState &session, std::uint64_t keep_positions) {
  const std::uint64_t cached = session.cached_token_count;
  if (keep_positions >= cached || cached == 0) {
    return;
  }
  for (SessionState::LayerKvCache &layer : session.layered_kv_cache) {
    const std::size_t lane_size = layer.keys.size();
    if (lane_size == 0 || lane_size % static_cast<std::size_t>(cached) != 0) {
      continue;
    }
    const std::size_t stride = lane_size / static_cast<std::size_t>(cached);
    layer.keys.truncate(static_cast<std::size_t>(keep_positions) * stride);
    layer.values.truncate(static_cast<std::size_t>(keep_positions) * stride);
  }
  if (session.sequence_token_ids.size() > static_cast<std::size_t>(keep_positions)) {
    session.sequence_token_ids.resize(static_cast<std::size_t>(keep_positions));
  }
  session.cached_token_count = keep_positions;
  session.gpu_kv_cache_generation = next_gpu_kv_cache_generation_value();
  session.gpu_kv_max_positions = 0;
  std::fill(session.gpu_kv_cache_active.begin(), session.gpu_kv_cache_active.end(), false);
  session.gpu_token_graph_layers.clear();
  session.gpu_token_graph_descriptors_ready = false;
}

// --- Speculative decoding, prompt-lookup drafts (enhancement Phase 2.2) ------
// Greedy decode can verify K guessed tokens with ONE batched forward pass:
// the batch reads every layer's weights once instead of once per token, which
// is the entire cost of memory-bound CPU decode. Drafts come from an n-gram
// lookup over the session's own context (no draft model): if the last 2-3
// tokens already occurred earlier, the tokens that followed then are proposed
// now. Repetitive continuations (code, lists, quotes, chat transcripts)
// accept most drafts; when no n-gram matches, decode proceeds token-by-token
// with zero overhead. Output is byte-identical to the sequential greedy loop
// because every emitted token is the argmax of exact logits.
// LIZARD_NATIVE_SPECULATIVE=1 enables the lane (opt-in until it passes the
// benchmark gate); LIZARD_NATIVE_SPECULATIVE_DRAFT bounds drafts per round.
static bool speculative_decode_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("LIZARD_NATIVE_SPECULATIVE");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}

static std::size_t speculative_draft_limit() {
  static const std::size_t limit = [] {
    const char *value = std::getenv("LIZARD_NATIVE_SPECULATIVE_DRAFT");
    if (value == nullptr) {
      return static_cast<std::size_t>(8);
    }
    const long parsed = std::strtol(value, nullptr, 10);
    return static_cast<std::size_t>(std::max(1l, std::min(32l, parsed)));
  }();
  return limit;
}

// Proposes up to `max_drafts` continuation tokens by matching the trailing
// n-gram of `context` (its last element is the token about to be fed) against
// earlier positions. Longest n-gram first, most recent occurrence wins.
static std::vector<std::uint64_t> draft_tokens_from_ngram(
  const std::vector<std::uint64_t> &context,
  std::size_t max_drafts) {
  std::vector<std::uint64_t> drafts;
  if (max_drafts == 0) {
    return drafts;
  }
  for (std::size_t ngram = 3; ngram >= 2; ngram -= 1) {
    if (context.size() < ngram + 1) {
      continue;
    }
    const std::size_t pattern_start = context.size() - ngram;
    // Most recent prior occurrence first: scan backwards, excluding the
    // pattern's own position.
    for (std::size_t start = pattern_start; start > 0;) {
      start -= 1;
      bool matches = true;
      for (std::size_t item = 0; item < ngram; item += 1) {
        if (context[start + item] != context[pattern_start + item]) {
          matches = false;
          break;
        }
      }
      if (!matches) {
        continue;
      }
      // Overlapping continuations are fine: drafts are only guesses and the
      // batched verification corrects any miss. continuation < context.size()
      // holds because start <= pattern_start - 1.
      const std::size_t continuation = start + ngram;
      const std::size_t available = context.size() - continuation;
      const std::size_t count = std::min(max_drafts, available);
      if (count == 0) {
        break;
      }
      drafts.assign(
        context.begin() + static_cast<std::ptrdiff_t>(continuation),
        context.begin() + static_cast<std::ptrdiff_t>(continuation + count));
      return drafts;
    }
  }
  return drafts;
}

static std::size_t resolve_gemma_kv_source_layer(
  const LoadedModelState &state,
  std::size_t layer_number,
  const GemmaLayerExecutionConfig &config) {
  const std::uint64_t shared_layer_count = state.model.attention_shared_kv_layers.value_or(0);
  const std::size_t layer_count = state.tensor_index.layers.size();
  if (shared_layer_count == 0) {
    return layer_number;
  }
  if (shared_layer_count >= layer_count) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Gemma shared-KV metadata leaves no source layers.");
  }
  const std::size_t first_shared_layer = layer_count - static_cast<std::size_t>(shared_layer_count);
  if (layer_number < first_shared_layer) {
    return layer_number;
  }

  const bool uses_swa_dimensions = state.model.attention_key_length_swa.has_value()
    && config.head_dim == state.model.attention_key_length_swa.value();
  const std::size_t source_distance = uses_swa_dimensions ? 2 : 1;
  if (first_shared_layer < source_distance) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Gemma shared-KV metadata does not provide a compatible source layer.");
  }
  return first_shared_layer - source_distance;
}

static void execute_llama_consume_token(
  const LoadedModelState &state,
  SessionState &session,
  std::uint64_t token_id) {
  const LlamaExecutionConfig *config = cached_llama_execution_config(state);
  if (config == nullptr) {
    throw WorkerError("NATIVE_EXECUTOR_UNIMPLEMENTED", "The loaded GGUF is missing the attention metadata required for native Llama execution.");
  }
  if (session.layered_kv_cache.size() != state.tensor_index.layers.size()) {
    session.layered_kv_cache.assign(state.tensor_index.layers.size(), SessionState::LayerKvCache {});
  }

  std::vector<float> hidden_state = tensor_row_f32(
    state,
    tensor_at(state, state.tensor_index.token_embeddings),
    token_id);
  const std::uint64_t position = session.cached_token_count;
  const std::uint64_t kv_hidden_size = config->kv_head_count * config->head_dim;

  for (std::size_t layer_number = 0; layer_number < state.tensor_index.layers.size(); layer_number += 1) {
    const LlamaLayerTensorIndex &layer_index = state.tensor_index.layers[layer_number];
    SessionState::LayerKvCache &layer_cache = session.layered_kv_cache[layer_number];

    // Norm weights are pre-dequantized into vector_tensor_cache at load;
    // fetching through the cache avoids two hidden_size-float heap copies
    // per layer per token that the uncached call used to pay.
    const std::vector<float> &attention_norm_weights = cached_tensor_vector_f32(state, layer_index.attention_norm, "Llama attention norm weights");
    const std::vector<float> &ffn_norm_weights = cached_tensor_vector_f32(state, layer_index.ffn_norm, "Llama FFN norm weights");

    const TensorInfo &query_tensor = tensor_at(state, layer_index.query);
    const TensorInfo &key_tensor = tensor_at(state, layer_index.key);
    const TensorInfo &value_tensor = tensor_at(state, layer_index.value);
    const TensorInfo &output_tensor = tensor_at(state, layer_index.output);
    const TensorInfo &gate_tensor = tensor_at(state, layer_index.gate);
    const TensorInfo &down_tensor = tensor_at(state, layer_index.down);
    const TensorInfo &up_tensor = tensor_at(state, layer_index.up);

    const bool shapes_cached = state.llama_layer_shapes_ready
      && layer_number < state.llama_layer_shapes.size();
    const LlamaLayerShapeConfig live_shapes = shapes_cached
      ? LlamaLayerShapeConfig {}
      : LlamaLayerShapeConfig {
          tensor_matrix_shape(query_tensor, "Llama attention query projection"),
          tensor_matrix_shape(key_tensor, "Llama attention key projection"),
          tensor_matrix_shape(value_tensor, "Llama attention value projection"),
          tensor_matrix_shape(output_tensor, "Llama attention output projection"),
          tensor_matrix_shape(gate_tensor, "Llama MLP gate projection"),
          tensor_matrix_shape(down_tensor, "Llama MLP down projection"),
          tensor_matrix_shape(up_tensor, "Llama MLP up projection"),
        };
    const LlamaLayerShapeConfig &shapes = shapes_cached
      ? state.llama_layer_shapes[layer_number]
      : live_shapes;
    const MatrixShape &query_shape = shapes.query_shape;
    const MatrixShape &key_shape = shapes.key_shape;
    const MatrixShape &value_shape = shapes.value_shape;
    const MatrixShape &output_shape = shapes.output_shape;
    const MatrixShape &gate_shape = shapes.gate_shape;
    const MatrixShape &down_shape = shapes.down_shape;
    const MatrixShape &up_shape = shapes.up_shape;

    const std::vector<float> normed = rms_norm(hidden_state, attention_norm_weights);
    const auto qkv_started_at = std::chrono::steady_clock::now();
    const std::uint64_t qkv_gpu_before = g_decode_gpu_use_count;
    const std::string qkv_detail_before = g_decode_gpu_detail;
    // One fused Q/K/V submission replaces three fence round-trips — and
    // brings llama's K/V projections onto the GPU lane at all: individually
    // they sit below the per-matvec GPU work threshold and ran on CPU.
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    bool fused_qkv = false;
    if (!session.decode_qkv_fuse_blocked) {
      const TensorInfo *qkv_tensors[3] = {&query_tensor, &key_tensor, &value_tensor};
      const MatrixShape *qkv_shapes[3] = {&query_shape, &key_shape, &value_shape};
      std::vector<float> *qkv_outputs[3] = {&query, &key, &value};
      fused_qkv = try_multi_matvec_resident(state, qkv_tensors, qkv_shapes, 3, normed, qkv_outputs);
      if (!fused_qkv) {
        session.decode_qkv_fuse_blocked = true;
      }
    }
    if (!fused_qkv) {
      const TensorInfo *qkv_tensors[3] = {&query_tensor, &key_tensor, &value_tensor};
      const MatrixShape *qkv_shapes[3] = {&query_shape, &key_shape, &value_shape};
      std::vector<float> *qkv_outputs[3] = {&query, &key, &value};
      fused_qkv = try_multi_matvec_pinned_ggml(
        state,
        qkv_tensors,
        qkv_shapes,
        3,
        normed,
        qkv_outputs,
        "Llama grouped QKV");
    }
    if (!fused_qkv) {
      query = matrix_vector_mul_tensor(
        state,
        query_tensor,
        query_shape,
        normed,
        "Llama attention query projection");
      key = matrix_vector_mul_tensor(
        state,
        key_tensor,
        key_shape,
        normed,
        "Llama attention key projection");
      value = matrix_vector_mul_tensor(
        state,
        value_tensor,
        value_shape,
        normed,
        "Llama attention value projection");
    }

    apply_rope_in_place(
      query,
      position,
      config->head_count,
      config->head_dim,
      config->rope_freq_base,
      config->rope_freq_scale,
      config->rope_freq_factors);
    apply_rope_in_place(
      key,
      position,
      config->kv_head_count,
      config->head_dim,
      config->rope_freq_base,
      config->rope_freq_scale,
      config->rope_freq_factors);
    record_decode_phase_metric(
      session,
      "Llama decode attention QKV",
      static_cast<std::int64_t>(layer_number),
      qkv_started_at,
      qkv_gpu_before,
      qkv_detail_before);

    append_layer_cache(layer_cache, key, value);
    const std::uint64_t cached_positions = session.cached_token_count + 1;
    if (layer_cache.keys.size() != static_cast<std::size_t>(cached_positions * kv_hidden_size)
      || layer_cache.values.size() != static_cast<std::size_t>(cached_positions * kv_hidden_size)) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Native Llama KV cache size is inconsistent with the current session state.");
    }

    const auto attention_cpu_started_at = std::chrono::steady_clock::now();
    const std::uint64_t attention_cpu_gpu_before = g_decode_gpu_use_count;
    const std::string attention_cpu_detail_before = g_decode_gpu_detail;
    std::vector<float> attention_context(static_cast<std::size_t>(config->hidden_size), 0.0f);
    const float attention_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(config->head_dim)));
    const auto compute_attention_heads = [&](std::uint64_t start_head, std::uint64_t end_head) {
      std::vector<float> kv_row_scratch;
      std::vector<float> scores;
      for (std::uint64_t head = start_head; head < end_head; head += 1) {
        const std::uint64_t kv_head = head * config->kv_head_count / config->head_count;
        const std::size_t query_offset = static_cast<std::size_t>(head * config->head_dim);
        const float *query_ptr = query.data() + static_cast<std::ptrdiff_t>(query_offset);
        scores.assign(static_cast<std::size_t>(cached_positions), 0.0f);
        for (std::uint64_t source = 0; source < cached_positions; source += 1) {
          float dot = 0.0f;
          const std::size_t source_base = static_cast<std::size_t>(source * kv_hidden_size + kv_head * config->head_dim);
          const float *key_ptr = layer_cache.keys.row_f32(
            source_base, static_cast<std::size_t>(config->head_dim), kv_row_scratch);
          dot += attention_dot(query_ptr, key_ptr, static_cast<std::size_t>(config->head_dim));
          scores[static_cast<std::size_t>(source)] = dot * attention_scale;
        }
        softmax_in_place(scores);
        for (std::uint64_t source = 0; source < cached_positions; source += 1) {
          const std::size_t source_base = static_cast<std::size_t>(source * kv_hidden_size + kv_head * config->head_dim);
          const float weight = scores[static_cast<std::size_t>(source)];
          const float *value_ptr = layer_cache.values.row_f32(
            source_base, static_cast<std::size_t>(config->head_dim), kv_row_scratch);
          float *context_ptr = attention_context.data() + static_cast<std::ptrdiff_t>(query_offset);
          attention_axpy(context_ptr, value_ptr, weight, static_cast<std::size_t>(config->head_dim));
        }
      }
    };
    const std::uint64_t attention_worker_count = choose_row_worker_count(
      config->head_count * cached_positions * config->head_dim,
      config->head_count,
      16384,
      1);
    if (attention_worker_count <= 1) {
      compute_attention_heads(0, config->head_count);
    } else {
      persistent_row_workers().execute(config->head_count, attention_worker_count, compute_attention_heads);
    }
    record_decode_phase_metric(
      session,
      "Llama decode attention CPU",
      static_cast<std::int64_t>(layer_number),
      attention_cpu_started_at,
      attention_cpu_gpu_before,
      attention_cpu_detail_before);

    // One-submission layer tail. Falls back to the split phases below for the
    // rest of the session on any refusal, so an unsupported weight format or a
    // non-finite result costs one attempt rather than the model.
    if (!session.decode_layer_tail_fuse_blocked) {
      const auto tail_started_at = std::chrono::steady_clock::now();
      const std::uint64_t tail_gpu_before = g_decode_gpu_use_count;
      const std::string tail_detail_before = g_decode_gpu_detail;
      std::vector<float> fused_hidden;
      if (try_llama_layer_tail_resident(
          state,
          output_tensor,
          output_shape,
          gate_tensor,
          gate_shape,
          up_tensor,
          up_shape,
          down_tensor,
          down_shape,
          attention_context,
          hidden_state,
          ffn_norm_weights,
          // Must match the split path's rms_norm below exactly, which takes the
          // 1e-5 default. A different epsilon here would make the fused and
          // unfused paths disagree by a hair and fail the token-identity gate.
          1e-5f,
          fused_hidden)) {
        record_decode_phase_metric(
          session,
          "Llama decode fused layer tail",
          static_cast<std::int64_t>(layer_number),
          tail_started_at,
          tail_gpu_before,
          tail_detail_before);
        hidden_state = std::move(fused_hidden);
        continue;
      }
      session.decode_layer_tail_fuse_blocked = true;
      session.decode_layer_tail_fuse_detail =
        "Direct3D 12 fused Llama decode layer tail fell back to split decode phases.";
    }

    const auto output_started_at = std::chrono::steady_clock::now();
    const std::uint64_t output_gpu_before = g_decode_gpu_use_count;
    const std::string output_detail_before = g_decode_gpu_detail;
    std::vector<float> residual = matrix_vector_mul_tensor(
      state,
      output_tensor,
      output_shape,
      attention_context,
      "Llama attention output projection");
    record_decode_phase_metric(
      session,
      "Llama decode attention output",
      static_cast<std::int64_t>(layer_number),
      output_started_at,
      output_gpu_before,
      output_detail_before);
    if (residual.size() != hidden_state.size()) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Llama attention residual vectors have mismatched sizes.");
    }
    for (std::size_t index = 0; index < residual.size(); index += 1) {
      residual[index] = hidden_state[index] + residual[index];
    }
    const std::vector<float> ffn_normed = rms_norm(residual, ffn_norm_weights);
    const auto ffn_started_at = std::chrono::steady_clock::now();
    const std::uint64_t ffn_gpu_before = g_decode_gpu_use_count;
    const std::string ffn_detail_before = g_decode_gpu_detail;
    // Whole-FFN single submission (gate+up matmuls, silu(gate)*up on device,
    // down matmul) replaces three fence round-trips and the CPU silu pass.
    std::vector<float> down;
    bool fused_ffn = false;
    if (!session.decode_ffn_fuse_blocked) {
      fused_ffn = try_ffn_chain_matvec_resident(
        state,
        gate_tensor,
        gate_shape,
        up_tensor,
        up_shape,
        down_tensor,
        down_shape,
        ffn_normed,
        down,
        true);
      if (!fused_ffn) {
        session.decode_ffn_fuse_blocked = true;
        session.decode_ffn_fuse_detail =
          "Direct3D 12 fused decode FFN fell back to split projections for this session.";
      }
    }
    if (!fused_ffn) {
      fused_ffn = try_ffn_swiglu_pinned_ggml(
        state,
        gate_tensor,
        gate_shape,
        up_tensor,
        up_shape,
        down_tensor,
        down_shape,
        ffn_normed,
        down);
    }
    if (!fused_ffn) {
      std::vector<float> gate;
      std::vector<float> up;
      const TensorInfo *gate_up_tensors[2] = {&gate_tensor, &up_tensor};
      const MatrixShape *gate_up_shapes[2] = {&gate_shape, &up_shape};
      std::vector<float> *gate_up_outputs[2] = {&gate, &up};
      const bool grouped_gate_up = try_multi_matvec_pinned_ggml(
        state,
        gate_up_tensors,
        gate_up_shapes,
        2,
        ffn_normed,
        gate_up_outputs,
        "Llama grouped gate/up");
      if (!grouped_gate_up) {
        gate = matrix_vector_mul_tensor(
          state,
          gate_tensor,
          gate_shape,
          ffn_normed,
          "Llama MLP gate projection");
        up = matrix_vector_mul_tensor(
          state,
          up_tensor,
          up_shape,
          ffn_normed,
          "Llama MLP up projection");
      }
      if (gate.size() != up.size()) {
        throw WorkerError("NATIVE_WORKER_ERROR", "Llama MLP gate and up projections produced mismatched shapes.");
      }
      for (std::size_t index = 0; index < gate.size(); index += 1) {
        gate[index] = silu(gate[index]) * up[index];
      }
      down = matrix_vector_mul_tensor(
        state,
        down_tensor,
        down_shape,
        gate,
        "Llama MLP down projection");
    }
    record_decode_phase_metric(
      session,
      "Llama decode FFN",
      static_cast<std::int64_t>(layer_number),
      ffn_started_at,
      ffn_gpu_before,
      ffn_detail_before);
    if (residual.size() != down.size()) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Llama MLP residual vectors have mismatched sizes.");
    }
    for (std::size_t index = 0; index < residual.size(); index += 1) {
      residual[index] += down[index];
    }
    hidden_state = std::move(residual);
  }

  session.sequence_token_ids.push_back(token_id);
  session.cached_token_count += 1;
  session.last_hidden_state = rms_norm(
    hidden_state,
    cached_tensor_vector_f32(state, state.tensor_index.output_norm, "Llama output norm weights"));
  session.last_logits = project_hidden_to_logits(
    state,
    tensor_at(state, state.tensor_index.output),
    session.last_hidden_state);
}

// Routed expert FFN for the MoE families (mixtral, qwen2moe, qwen3moe,
// deepseek). Softmax gating over all experts, top-k selection, optional
// top-k weight renormalization, optional routed-weight scale, plus the
// family's shared-expert branch.
static std::vector<float> execute_family_moe_ffn(
  const LoadedModelState &state,
  const FamilyExecutionConfig &config,
  const LlamaLayerTensorIndex &layer_index,
  const std::vector<float> &input) {
  const FamilySpec &spec = config.spec;
  const TensorInfo &gate_inp_tensor = tensor_at(state, layer_index.moe_gate_inp);
  const TensorInfo &gate_exps_tensor = tensor_at(state, layer_index.moe_gate_exps);
  const TensorInfo &up_exps_tensor = tensor_at(state, layer_index.moe_up_exps);
  const TensorInfo &down_exps_tensor = tensor_at(state, layer_index.moe_down_exps);
  if (gate_exps_tensor.dimensions.size() < 3
    || up_exps_tensor.dimensions.size() < 3
    || down_exps_tensor.dimensions.size() < 3) {
    throw WorkerError("NATIVE_WORKER_ERROR", "MoE expert tensors require three dimensions (width, rows, experts).");
  }
  const std::uint64_t expert_count = gate_exps_tensor.dimensions[2];
  const std::uint64_t expert_ff = gate_exps_tensor.dimensions[1];
  if (expert_count == 0 || expert_ff == 0 || expert_count != config.expert_count) {
    throw WorkerError("NATIVE_WORKER_ERROR", "MoE expert tensor layout does not match the GGUF expert metadata.");
  }
  if (gate_exps_tensor.dimensions[0] != config.hidden_size
    || up_exps_tensor.dimensions[0] != config.hidden_size
    || up_exps_tensor.dimensions[1] != expert_ff
    || down_exps_tensor.dimensions[0] != expert_ff
    || down_exps_tensor.dimensions[1] != config.hidden_size) {
    throw WorkerError("NATIVE_WORKER_ERROR", "MoE expert tensor shapes are inconsistent with the model dimensions.");
  }

  const MatrixShape gate_inp_shape = tensor_matrix_shape(gate_inp_tensor, "MoE router projection");
  if (gate_inp_shape.row_width != config.hidden_size || gate_inp_shape.row_count != expert_count) {
    throw WorkerError("NATIVE_WORKER_ERROR", "MoE router tensor shape does not match the expert count.");
  }
  const std::vector<float> router_logits = matrix_vector_mul_tensor(
    state, gate_inp_tensor, gate_inp_shape, input, "MoE router projection");
  const std::vector<float> router_probs = softmax(router_logits);

  const std::uint64_t used = std::min<std::uint64_t>(config.expert_used_count, expert_count);
  std::vector<std::size_t> expert_order(static_cast<std::size_t>(expert_count));
  for (std::size_t index = 0; index < expert_order.size(); index += 1) {
    expert_order[index] = index;
  }
  std::partial_sort(
    expert_order.begin(),
    expert_order.begin() + static_cast<std::ptrdiff_t>(used),
    expert_order.end(),
    [&](std::size_t left, std::size_t right) {
      return router_probs[left] > router_probs[right];
    });

  std::vector<float> weights(static_cast<std::size_t>(used), 0.0f);
  double weight_sum = 0.0;
  for (std::uint64_t slot = 0; slot < used; slot += 1) {
    weights[static_cast<std::size_t>(slot)] = router_probs[expert_order[static_cast<std::size_t>(slot)]];
    weight_sum += static_cast<double>(weights[static_cast<std::size_t>(slot)]);
  }
  if (spec.moe_norm_topk_weights && weight_sum > 0.0) {
    for (float &weight : weights) {
      weight = static_cast<float>(static_cast<double>(weight) / weight_sum);
    }
  }
  if (config.expert_weights_scale != 0.0 && config.expert_weights_scale != 1.0) {
    for (float &weight : weights) {
      weight = static_cast<float>(static_cast<double>(weight) * config.expert_weights_scale);
    }
  }

  std::vector<float> moe_output(static_cast<std::size_t>(config.hidden_size), 0.0f);
  for (std::uint64_t slot = 0; slot < used; slot += 1) {
    const std::uint64_t expert = static_cast<std::uint64_t>(expert_order[static_cast<std::size_t>(slot)]);
    const std::vector<float> expert_gate = matrix_vector_mul_tensor_rows(
      state, gate_exps_tensor, expert * expert_ff, expert_ff, config.hidden_size, input, "MoE expert gate projection");
    const std::vector<float> expert_up = matrix_vector_mul_tensor_rows(
      state, up_exps_tensor, expert * expert_ff, expert_ff, config.hidden_size, input, "MoE expert up projection");
    std::vector<float> activated(expert_gate.size(), 0.0f);
    for (std::size_t index = 0; index < expert_gate.size(); index += 1) {
      activated[index] = silu(expert_gate[index]) * expert_up[index];
    }
    const std::vector<float> expert_down = matrix_vector_mul_tensor_rows(
      state, down_exps_tensor, expert * config.hidden_size, config.hidden_size, expert_ff, activated, "MoE expert down projection");
    const float weight = weights[static_cast<std::size_t>(slot)];
    for (std::size_t index = 0; index < moe_output.size(); index += 1) {
      moe_output[index] += weight * expert_down[index];
    }
  }

  if (spec.moe_shared_expert
    && layer_index.shexp_gate != k_missing_tensor_index
    && layer_index.shexp_up != k_missing_tensor_index
    && layer_index.shexp_down != k_missing_tensor_index) {
    const TensorInfo &shexp_gate_tensor = tensor_at(state, layer_index.shexp_gate);
    const TensorInfo &shexp_up_tensor = tensor_at(state, layer_index.shexp_up);
    const TensorInfo &shexp_down_tensor = tensor_at(state, layer_index.shexp_down);
    const MatrixShape shexp_gate_shape = tensor_matrix_shape(shexp_gate_tensor, "Shared expert gate projection");
    const MatrixShape shexp_up_shape = tensor_matrix_shape(shexp_up_tensor, "Shared expert up projection");
    const MatrixShape shexp_down_shape = tensor_matrix_shape(shexp_down_tensor, "Shared expert down projection");
    const std::vector<float> shexp_gate = matrix_vector_mul_tensor(
      state, shexp_gate_tensor, shexp_gate_shape, input, "Shared expert gate projection");
    const std::vector<float> shexp_up = matrix_vector_mul_tensor(
      state, shexp_up_tensor, shexp_up_shape, input, "Shared expert up projection");
    if (shexp_gate.size() != shexp_up.size()) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Shared expert gate and up projections produced mismatched shapes.");
    }
    std::vector<float> shexp_activated(shexp_gate.size(), 0.0f);
    for (std::size_t index = 0; index < shexp_gate.size(); index += 1) {
      shexp_activated[index] = silu(shexp_gate[index]) * shexp_up[index];
    }
    std::vector<float> shexp_output = matrix_vector_mul_tensor(
      state, shexp_down_tensor, shexp_down_shape, shexp_activated, "Shared expert down projection");
    if (spec.moe_shared_expert_gated && layer_index.shexp_gate_inp != k_missing_tensor_index) {
      const std::vector<float> &shexp_gate_inp = cached_tensor_vector_f32(
        state, layer_index.shexp_gate_inp, "Shared expert router gate");
      if (shexp_gate_inp.size() != input.size()) {
        throw WorkerError("NATIVE_WORKER_ERROR", "Shared expert router gate size does not match the hidden size.");
      }
      double gate_logit = 0.0;
      for (std::size_t index = 0; index < input.size(); index += 1) {
        gate_logit += static_cast<double>(shexp_gate_inp[index]) * static_cast<double>(input[index]);
      }
      const float gate_value = static_cast<float>(1.0 / (1.0 + std::exp(-gate_logit)));
      for (float &value : shexp_output) {
        value *= gate_value;
      }
    }
    for (std::size_t index = 0; index < moe_output.size(); index += 1) {
      moe_output[index] += shexp_output[index];
    }
  }
  return moe_output;
}

// Parameterized per-token forward pass for the GGUF family expansion. The
// llama and gemma consume functions above stay byte-for-byte untouched;
// models with a FamilySpec route here instead.
static void execute_family_consume_token(
  const LoadedModelState &state,
  SessionState &session,
  std::uint64_t token_id) {
  const auto config_opt = resolve_family_execution_config(state);
  if (!config_opt.has_value()) {
    throw WorkerError("NATIVE_EXECUTOR_UNIMPLEMENTED", "The loaded GGUF is missing the attention metadata required for native family execution.");
  }
  const FamilyExecutionConfig &config = config_opt.value();
  const FamilySpec &spec = config.spec;
  if (session.layered_kv_cache.size() != state.tensor_index.layers.size()) {
    session.layered_kv_cache.assign(state.tensor_index.layers.size(), SessionState::LayerKvCache {});
  }

  const auto optional_vector = [&](std::size_t index, const char *label) -> const std::vector<float> * {
    if (index == k_missing_tensor_index) {
      return nullptr;
    }
    return &cached_tensor_vector_f32(state, index, label);
  };
  const auto apply_norm = [&](const std::vector<float> &input, std::size_t weight_index, std::size_t bias_index, const char *label) {
    const std::vector<float> &weights = cached_tensor_vector_f32(state, weight_index, label);
    if (spec.norm_kind == FamilyNormKind::layer) {
      return layer_norm(input, weights, optional_vector(bias_index, label), config.norm_epsilon);
    }
    return rms_norm(input, weights, config.norm_epsilon);
  };

  std::vector<float> hidden_state = tensor_row_f32(
    state,
    tensor_at(state, state.tensor_index.token_embeddings),
    token_id);
  const std::uint64_t position = session.cached_token_count;
  const std::uint64_t query_size = config.head_count * config.head_dim;
  const std::uint64_t kv_hidden_size = config.kv_head_count * config.head_dim;

  for (std::size_t layer_number = 0; layer_number < state.tensor_index.layers.size(); layer_number += 1) {
    const LlamaLayerTensorIndex &layer_index = state.tensor_index.layers[layer_number];
    SessionState::LayerKvCache &layer_cache = session.layered_kv_cache[layer_number];

    const std::vector<float> normed = apply_norm(
      hidden_state, layer_index.attention_norm, layer_index.attention_norm_bias, "Family attention norm");

    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    const auto qkv_phase_start = std::chrono::steady_clock::now();
    if (layer_index.qkv_fused != k_missing_tensor_index) {
      const TensorInfo &qkv_tensor = tensor_at(state, layer_index.qkv_fused);
      const MatrixShape qkv_shape = tensor_matrix_shape(qkv_tensor, "Family fused QKV projection");
      if (qkv_shape.row_width != config.hidden_size
        || qkv_shape.row_count != query_size + 2 * kv_hidden_size) {
        throw WorkerError("NATIVE_WORKER_ERROR", "Family fused QKV projection shape does not match the attention dimensions.");
      }
      std::vector<float> fused = matrix_vector_mul_tensor(
        state, qkv_tensor, qkv_shape, normed, "Family fused QKV projection");
      if (const std::vector<float> *fused_bias = optional_vector(layer_index.qkv_fused_bias, "Family fused QKV bias")) {
        add_bias_in_place(fused, *fused_bias, "Family fused QKV projection");
      }
      query.assign(fused.begin(), fused.begin() + static_cast<std::ptrdiff_t>(query_size));
      key.assign(
        fused.begin() + static_cast<std::ptrdiff_t>(query_size),
        fused.begin() + static_cast<std::ptrdiff_t>(query_size + kv_hidden_size));
      value.assign(
        fused.begin() + static_cast<std::ptrdiff_t>(query_size + kv_hidden_size),
        fused.end());
    } else {
      const TensorInfo &query_tensor = tensor_at(state, layer_index.query);
      const TensorInfo &key_tensor = tensor_at(state, layer_index.key);
      const TensorInfo &value_tensor = tensor_at(state, layer_index.value);
      const MatrixShape query_shape = tensor_matrix_shape(query_tensor, "Family attention query projection");
      const MatrixShape key_shape = tensor_matrix_shape(key_tensor, "Family attention key projection");
      const MatrixShape value_shape = tensor_matrix_shape(value_tensor, "Family attention value projection");
      if (query_shape.row_width != config.hidden_size || query_shape.row_count != query_size
        || key_shape.row_width != config.hidden_size || key_shape.row_count != kv_hidden_size
        || value_shape.row_width != config.hidden_size || value_shape.row_count != kv_hidden_size) {
        throw WorkerError("NATIVE_WORKER_ERROR", "Family attention projection shapes do not match the attention dimensions.");
      }
      // Q, K and V in ONE submission instead of three. All three read the same
      // normed hidden state and write disjoint outputs, so they are the same
      // shape of work the FFN gate/up pair already batches.
      //
      // Why it matters more than the arithmetic suggests: measured on this
      // iGPU, a submission costs ~0.37 ms of fence/submit latency beyond the
      // GPU time inside it. Three separate submissions per layer is 84 per
      // token on a 28-layer model - about 31 ms of pure latency against the
      // ~15 ms of real weight traffic these three matvecs represent. The
      // per-stage GPU timestamps say the kernels are already at bus bandwidth
      // (16.5 GB/s aggregate, gate+up at 20.5), so submission count is what is
      // left to attack.
      bool qkv_batched = false;
      if (gpu_decode_lane_enabled_for_model(state)) {
        D3d12WeightFormat query_format = D3d12WeightFormat::F32;
        D3d12WeightFormat key_format = D3d12WeightFormat::F32;
        D3d12WeightFormat value_format = D3d12WeightFormat::F32;
        if (d3d12_weight_format_supported(query_tensor.ggml_type, query_shape.row_width, &query_format)
          && d3d12_weight_format_supported(key_tensor.ggml_type, key_shape.row_width, &key_format)
          && d3d12_weight_format_supported(value_tensor.ggml_type, value_shape.row_width, &value_format)) {
          const std::uint8_t *const query_data = tensor_data_ptr(state, query_tensor);
          const std::uint8_t *const key_data = tensor_data_ptr(state, key_tensor);
          const std::uint8_t *const value_data = tensor_data_ptr(state, value_tensor);
          D3d12ResidentWeightRef ops[3];
          const TensorInfo *const tensors[3] = {&query_tensor, &key_tensor, &value_tensor};
          const MatrixShape *const shapes[3] = {&query_shape, &key_shape, &value_shape};
          const D3d12WeightFormat formats[3] = {query_format, key_format, value_format};
          const std::uint8_t *const payloads[3] = {query_data, key_data, value_data};
          for (std::size_t index = 0; index < 3; index += 1) {
            ops[index].weight_key = payloads[index];
            ops[index].format = formats[index];
            ops[index].weight_data = payloads[index];
            ops[index].weight_bytes = static_cast<std::size_t>(tensors[index]->byte_length);
            ops[index].row_count = static_cast<std::uint32_t>(shapes[index]->row_count);
            ops[index].row_width = static_cast<std::uint32_t>(shapes[index]->row_width);
          }
          std::vector<float> flat_outputs[3];
          if (d3d12_batch_matmul_resident(
            ops, 3, normed, 1, next_gpu_input_identity(), flat_outputs, nullptr)
            && flat_outputs[0].size() == query_shape.row_count
            && flat_outputs[1].size() == key_shape.row_count
            && flat_outputs[2].size() == value_shape.row_count) {
            mark_decode_gpu_used(
              "Direct3D 12 resident decode QKV triple executed on "
              + d3d12_batch_matmul_capabilities().adapter_name + ".");
            query = std::move(flat_outputs[0]);
            key = std::move(flat_outputs[1]);
            value = std::move(flat_outputs[2]);
            qkv_batched = true;
          }
        }
      }
      // Any refusal - unsupported quant, no resident capacity, GPU lane off -
      // falls back to the three individual matvecs, which each pick their own
      // GPU or CPU lane exactly as before.
      if (!qkv_batched) {
        query = matrix_vector_mul_tensor(state, query_tensor, query_shape, normed, "Family attention query projection");
        key = matrix_vector_mul_tensor(state, key_tensor, key_shape, normed, "Family attention key projection");
        value = matrix_vector_mul_tensor(state, value_tensor, value_shape, normed, "Family attention value projection");
      }
      if (const std::vector<float> *query_bias = optional_vector(layer_index.query_bias, "Family attention query bias")) {
        add_bias_in_place(query, *query_bias, "Family attention query projection");
      }
      if (const std::vector<float> *key_bias = optional_vector(layer_index.key_bias, "Family attention key bias")) {
        add_bias_in_place(key, *key_bias, "Family attention key projection");
      }
      if (const std::vector<float> *value_bias = optional_vector(layer_index.value_bias, "Family attention value bias")) {
        add_bias_in_place(value, *value_bias, "Family attention value projection");
      }
    }
    record_decode_phase_since("qkv-projection", qkv_phase_start);

    const auto rope_phase_start = std::chrono::steady_clock::now();
    if (spec.qk_norm) {
      query = rms_norm_heads(
        query,
        cached_tensor_vector_f32(state, layer_index.query_norm, "Family attention query norm"),
        config.head_count,
        config.head_dim,
        "Family attention query norm",
        config.norm_epsilon);
      key = rms_norm_heads(
        key,
        cached_tensor_vector_f32(state, layer_index.key_norm, "Family attention key norm"),
        config.kv_head_count,
        config.head_dim,
        "Family attention key norm",
        config.norm_epsilon);
    }

    apply_family_rope_in_place(
      query,
      position,
      config.head_count,
      config.head_dim,
      config.rope_dim,
      spec.rope_kind,
      config.rope_freq_base,
      config.rope_freq_scale,
      config.rope_freq_factors,
      config.rope_attn_factor);
    apply_family_rope_in_place(
      key,
      position,
      config.kv_head_count,
      config.head_dim,
      config.rope_dim,
      spec.rope_kind,
      config.rope_freq_base,
      config.rope_freq_scale,
      config.rope_freq_factors,
      config.rope_attn_factor);
    record_decode_phase_since("rope+qknorm", rope_phase_start);

    const auto kv_append_start = std::chrono::steady_clock::now();
    append_layer_cache(layer_cache, key, value);
    const std::uint64_t cached_positions = session.cached_token_count + 1;
    if (layer_cache.keys.size() != static_cast<std::size_t>(cached_positions * kv_hidden_size)
      || layer_cache.values.size() != static_cast<std::size_t>(cached_positions * kv_hidden_size)) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Native family KV cache size is inconsistent with the current session state.");
    }
    record_decode_phase_since("kv-append", kv_append_start);

    const auto attention_phase_start = std::chrono::steady_clock::now();
    std::vector<float> attention_context(static_cast<std::size_t>(query_size), 0.0f);
    const float attention_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(config.head_dim)));
    const auto compute_attention_heads = [&](std::uint64_t start_head, std::uint64_t end_head) {
      std::vector<float> kv_row_scratch;
      for (std::uint64_t head = start_head; head < end_head; head += 1) {
        const std::uint64_t kv_head = head * config.kv_head_count / config.head_count;
        const std::size_t query_offset = static_cast<std::size_t>(head * config.head_dim);
        const float *query_ptr = query.data() + static_cast<std::ptrdiff_t>(query_offset);
        std::vector<float> scores(static_cast<std::size_t>(cached_positions), 0.0f);
        for (std::uint64_t source = 0; source < cached_positions; source += 1) {
          float dot = 0.0f;
          const std::size_t source_base = static_cast<std::size_t>(source * kv_hidden_size + kv_head * config.head_dim);
          const float *key_ptr = layer_cache.keys.row_f32(
            source_base, static_cast<std::size_t>(config.head_dim), kv_row_scratch);
          dot += attention_dot(query_ptr, key_ptr, static_cast<std::size_t>(config.head_dim));
          scores[static_cast<std::size_t>(source)] = dot * attention_scale;
        }
        const std::vector<float> attention_weights = softmax(scores);
        for (std::uint64_t source = 0; source < cached_positions; source += 1) {
          const std::size_t source_base = static_cast<std::size_t>(source * kv_hidden_size + kv_head * config.head_dim);
          const float weight = attention_weights[static_cast<std::size_t>(source)];
          const float *value_ptr = layer_cache.values.row_f32(
            source_base, static_cast<std::size_t>(config.head_dim), kv_row_scratch);
          float *context_ptr = attention_context.data() + static_cast<std::ptrdiff_t>(query_offset);
          attention_axpy(context_ptr, value_ptr, weight, static_cast<std::size_t>(config.head_dim));
        }
      }
    };
    const std::uint64_t attention_worker_count = choose_row_worker_count(
      config.head_count * cached_positions * config.head_dim,
      config.head_count,
      16384,
      1);
    if (attention_worker_count <= 1) {
      compute_attention_heads(0, config.head_count);
    } else {
      persistent_row_workers().execute(config.head_count, attention_worker_count, compute_attention_heads);
    }
    record_decode_phase_since("attention-core", attention_phase_start);

    const TensorInfo &output_tensor = tensor_at(state, layer_index.output);
    const MatrixShape output_shape = tensor_matrix_shape(output_tensor, "Family attention output projection");
    if (output_shape.row_width != query_size || output_shape.row_count != config.hidden_size) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Family attention output projection shape does not match the attention dimensions.");
    }
    std::vector<float> attention_output = matrix_vector_mul_tensor(
      state, output_tensor, output_shape, attention_context, "Family attention output projection");
    if (const std::vector<float> *output_bias = optional_vector(layer_index.output_bias, "Family attention output bias")) {
      add_bias_in_place(attention_output, *output_bias, "Family attention output projection");
    }
    if (spec.post_attention_norm) {
      attention_output = rms_norm(
        attention_output,
        cached_tensor_vector_f32(state, layer_index.post_attention_norm, "Family post-attention norm"),
        config.norm_epsilon);
    }

    const auto run_dense_ffn = [&](const std::vector<float> &ffn_input) {
      const TensorInfo &down_tensor = tensor_at(state, layer_index.down);
      const TensorInfo &up_tensor = tensor_at(state, layer_index.up);
      const MatrixShape down_shape = tensor_matrix_shape(down_tensor, "Family MLP down projection");
      const MatrixShape up_shape = tensor_matrix_shape(up_tensor, "Family MLP up projection");
      std::vector<float> activated;
      if (spec.ffn_kind == FamilyFfnKind::swiglu_split) {
        const TensorInfo &gate_tensor = tensor_at(state, layer_index.gate);
        const MatrixShape gate_shape = tensor_matrix_shape(gate_tensor, "Family MLP gate projection");
        const std::vector<float> gate = matrix_vector_mul_tensor(
          state, gate_tensor, gate_shape, ffn_input, "Family MLP gate projection");
        const std::vector<float> up = matrix_vector_mul_tensor(
          state, up_tensor, up_shape, ffn_input, "Family MLP up projection");
        if (gate.size() != up.size()) {
          throw WorkerError("NATIVE_WORKER_ERROR", "Family MLP gate and up projections produced mismatched shapes.");
        }
        activated.assign(gate.size(), 0.0f);
        for (std::size_t index = 0; index < gate.size(); index += 1) {
          activated[index] = silu(gate[index]) * up[index];
        }
      } else if (spec.ffn_kind == FamilyFfnKind::swiglu_fused) {
        // Fused gate|up projection (phi3/glm4): the first half of the rows is
        // the gate, the second half is the up projection.
        if (up_shape.row_count % 2 != 0) {
          throw WorkerError("NATIVE_WORKER_ERROR", "Family fused MLP projection requires an even row count.");
        }
        std::vector<float> fused = matrix_vector_mul_tensor(
          state, up_tensor, up_shape, ffn_input, "Family fused MLP projection");
        const std::size_t half = fused.size() / 2;
        activated.assign(half, 0.0f);
        for (std::size_t index = 0; index < half; index += 1) {
          activated[index] = silu(fused[index]) * fused[half + index];
        }
      } else {
        std::vector<float> up = matrix_vector_mul_tensor(
          state, up_tensor, up_shape, ffn_input, "Family MLP up projection");
        if (const std::vector<float> *up_bias = optional_vector(layer_index.up_bias, "Family MLP up bias")) {
          add_bias_in_place(up, *up_bias, "Family MLP up projection");
        }
        activated.assign(up.size(), 0.0f);
        for (std::size_t index = 0; index < up.size(); index += 1) {
          activated[index] = gelu_tanh(up[index]);
        }
      }
      if (down_shape.row_width != activated.size() || down_shape.row_count != config.hidden_size) {
        throw WorkerError("NATIVE_WORKER_ERROR", "Family MLP down projection shape does not match the activation size.");
      }
      std::vector<float> down = matrix_vector_mul_tensor(
        state, down_tensor, down_shape, activated, "Family MLP down projection");
      if (const std::vector<float> *down_bias = optional_vector(layer_index.down_bias, "Family MLP down bias")) {
        add_bias_in_place(down, *down_bias, "Family MLP down projection");
      }
      return down;
    };

    if (spec.parallel_residual) {
      // phi2: attention and FFN both consume the pre-norm output; the layer
      // result is input + attention + FFN.
      const std::vector<float> ffn_output = run_dense_ffn(normed);
      std::vector<float> next_hidden(hidden_state.size(), 0.0f);
      for (std::size_t index = 0; index < hidden_state.size(); index += 1) {
        next_hidden[index] = hidden_state[index] + attention_output[index] + ffn_output[index];
      }
      hidden_state = std::move(next_hidden);
      continue;
    }

    const std::vector<float> residual = add_vectors(hidden_state, attention_output, "Family attention residual");
    const std::vector<float> ffn_normed = spec.has_ffn_norm
      ? apply_norm(residual, layer_index.ffn_norm, layer_index.ffn_norm_bias, "Family FFN norm")
      : residual;
    const bool moe_layer = spec.moe && layer_index.moe_gate_exps != k_missing_tensor_index;
    std::vector<float> ffn_output = moe_layer
      ? execute_family_moe_ffn(state, config, layer_index, ffn_normed)
      : run_dense_ffn(ffn_normed);
    if (spec.post_ffw_norm) {
      ffn_output = rms_norm(
        ffn_output,
        cached_tensor_vector_f32(state, layer_index.post_ffw_norm, "Family post-FFN norm"),
        config.norm_epsilon);
    }
    hidden_state = add_vectors(residual, ffn_output, "Family MLP residual");
  }

  session.sequence_token_ids.push_back(token_id);
  session.cached_token_count += 1;
  if (spec.norm_kind == FamilyNormKind::layer) {
    const std::vector<float> &output_norm_weights = cached_tensor_vector_f32(
      state, state.tensor_index.output_norm, "Family output norm");
    const std::vector<float> *output_norm_bias = state.tensor_index.output_norm_bias != k_missing_tensor_index
      ? &cached_tensor_vector_f32(state, state.tensor_index.output_norm_bias, "Family output norm bias")
      : nullptr;
    session.last_hidden_state = layer_norm(hidden_state, output_norm_weights, output_norm_bias, config.norm_epsilon);
  } else {
    session.last_hidden_state = rms_norm(
      hidden_state,
      cached_tensor_vector_f32(state, state.tensor_index.output_norm, "Family output norm"),
      config.norm_epsilon);
  }
  session.last_logits = project_hidden_to_logits(
    state,
    tensor_at(state, state.tensor_index.output),
    session.last_hidden_state);
  if (state.tensor_index.output_bias != k_missing_tensor_index) {
    add_bias_in_place(
      session.last_logits,
      cached_tensor_vector_f32(state, state.tensor_index.output_bias, "Family LM head bias"),
      "Family LM head projection");
  }
}

static bool try_execute_gemma_token_graph(
  const LoadedModelState &state,
  SessionState &session,
  std::uint64_t token_id,
  const std::vector<float> &initial_hidden,
  std::vector<float> &final_hidden,
  std::vector<float> &final_logits,
  bool need_logits,
  std::uint32_t &selected_token_out,
  std::string &error_message) {
  if (!gpu_decode_lane_enabled_for_model(state)
    || !gpu_gemma_attention_graph_enabled()
    || !gpu_gemma_token_graph_enabled()
    // f32/f16 lanes seed the device KV cache directly (it stores halves
    // natively). Quantized lanes have no upload path yet and stay on the
    // trusted CPU attention lanes.
    || kv_cache_mode_is_quantized()
    || !state.gemma_layer_configs_ready
    || state.gemma_layer_configs.size() != state.tensor_index.layers.size()) {
    return false;
  }
  if (session.gpu_kv_cache_generation == 0) {
    session.gpu_kv_cache_generation = next_gpu_kv_cache_generation_value();
  }
  if (session.gpu_kv_cache_active.size() != state.tensor_index.layers.size()) {
    session.gpu_kv_cache_active.assign(state.tensor_index.layers.size(), false);
  }
  const std::uint64_t position = session.cached_token_count;
  const std::uint64_t requested_cache_horizon =
    position + std::max<std::uint64_t>(session.max_tokens, 1) + 2;
  if (session.gpu_kv_max_positions == 0) {
    session.gpu_kv_max_positions = std::min<std::uint64_t>(
      state.model.context_length.value_or(requested_cache_horizon),
      requested_cache_horizon);
  }

  const std::uint64_t per_layer_input_size =
    state.model.embedding_length_per_layer_input.value_or(0);
  // Layer refs, PLE config, and the LM-head ref are session-invariant; only
  // attention.position and the per-token PLE inputs change per token.
  // Rebuilding all ~400 resident refs per token cost ~10 ms on the Iris Xe
  // baseline, so build once per KV-cache generation and patch per token.
  const bool rebuild_descriptors = !session.gpu_token_graph_descriptors_ready
    || session.gpu_token_graph_descriptor_generation != session.gpu_kv_cache_generation;
  std::vector<D3d12GemmaTokenLayer> &gpu_layers = session.gpu_token_graph_layers;
  D3d12GemmaPleInputConfig &gpu_ple_inputs = session.gpu_token_graph_ple;
  std::vector<float> scaled_ple_token_inputs;
  const D3d12GemmaPleInputConfig *gpu_ple_inputs_ptr = nullptr;
  if (rebuild_descriptors) {
  session.gpu_token_graph_descriptors_ready = false;
  gpu_layers.clear();
  gpu_layers.reserve(state.tensor_index.layers.size());
  gpu_ple_inputs = D3d12GemmaPleInputConfig {};
  if (per_layer_input_size > 0) {
    const TensorInfo &ple_model_projection =
      tensor_at(state, state.tensor_index.per_layer_model_projection);
    const MatrixShape ple_projection_shape = tensor_matrix_shape(
      ple_model_projection, "Gemma packed per-layer model projection");
    if (!build_resident_ref(
      state,
      ple_model_projection,
      ple_projection_shape,
      gpu_ple_inputs.model_projection)) {
      error_message = "Gemma token graph packed PLE projection is unsupported.";
      return false;
    }
    gpu_ple_inputs.projection_norm = &cached_tensor_vector_f32(
      state,
      state.tensor_index.per_layer_projection_norm,
      "Gemma per-layer projection norm");
    gpu_ple_inputs.layer_width =
      static_cast<std::uint32_t>(per_layer_input_size);
    gpu_ple_inputs.rms_epsilon =
      static_cast<float>(state.model.layer_norm_rms_epsilon.value_or(1e-5));
    gpu_ple_inputs.projection_scale = static_cast<float>(
      1.0 / std::sqrt(static_cast<double>(initial_hidden.size())));
  }
  for (std::size_t layer_number = 0;
    layer_number < state.tensor_index.layers.size();
    layer_number += 1) {
    const LlamaLayerTensorIndex &layer_index =
      state.tensor_index.layers[layer_number];
    const GemmaLayerExecutionConfig &config =
      state.gemma_layer_configs[layer_number];
    const std::size_t kv_source_layer =
      resolve_gemma_kv_source_layer(state, layer_number, config);
    if (kv_source_layer >= state.gemma_layer_configs.size()) {
      error_message = "Gemma token graph shared-KV source is out of range.";
      return false;
    }
    const GemmaLayerExecutionConfig &kv_config =
      state.gemma_layer_configs[kv_source_layer];
    if (kv_config.kv_size != config.kv_size
      || kv_config.head_dim != config.head_dim) {
      error_message = "Gemma token graph shared-KV dimensions are inconsistent.";
      return false;
    }

    D3d12GemmaTokenLayer layer;
    layer.owns_kv_cache = kv_source_layer == layer_number;
    if (!build_resident_ref(
        state,
        tensor_at(state, layer_index.query),
        config.query_shape,
        layer.query)
      || (layer.owns_kv_cache
        && (!build_resident_ref(
            state,
            tensor_at(state, layer_index.key),
            config.key_shape,
            layer.key)
          || !build_resident_ref(
            state,
            tensor_at(state, layer_index.value),
            config.value_shape,
            layer.value)))
      || !build_resident_ref(
        state,
        tensor_at(state, layer_index.output),
        config.output_shape,
        layer.attn_out)
      || !build_resident_ref(
        state,
        tensor_at(state, layer_index.gate),
        config.gate_shape,
        layer.gate)
      || !build_resident_ref(
        state,
        tensor_at(state, layer_index.up),
        config.up_shape,
        layer.up)
      || !build_resident_ref(
        state,
        tensor_at(state, layer_index.down),
        config.down_shape,
        layer.down)) {
      error_message = "Gemma token graph contains an unsupported core tensor format.";
      return false;
    }

    layer.attention_norm = &cached_tensor_vector_f32(
      state, layer_index.attention_norm, "Gemma attention norm");
    layer.query_norm = &cached_tensor_vector_f32(
      state, layer_index.query_norm, "Gemma query norm");
    layer.key_norm = &cached_tensor_vector_f32(
      state, layer_index.key_norm, "Gemma key norm");
    layer.rope_freq_factors = &config.rope_freq_factors;
    layer.post_attention_norm = &cached_tensor_vector_f32(
      state, layer_index.post_attention_norm, "Gemma post-attention norm");
    layer.ffn_norm = &cached_tensor_vector_f32(
      state, layer_index.ffn_norm, "Gemma FFN norm");
    layer.post_ffw_norm = &cached_tensor_vector_f32(
      state, layer_index.post_ffw_norm, "Gemma post-FFW norm");
    SessionState::LayerKvCache &layer_cache =
      session.layered_kv_cache[kv_source_layer];
    // f32 and f16 lanes seed the device cache directly (the device KV store
    // is f16 natively); the builder is gated so q8_0 lanes never reach here.
    layer.initial_keys = layer_cache.keys.gpu_lane_view();
    layer.initial_values = layer_cache.values.gpu_lane_view();

    if (config.per_layer_input_size > 0) {
      if (config.per_layer_input_size != per_layer_input_size
        || !build_resident_ref(
          state,
          tensor_at(state, layer_index.per_layer_input_gate),
          config.per_layer_input_gate_shape,
          layer.ple_gate)
        || !build_resident_ref(
          state,
          tensor_at(state, layer_index.per_layer_projection),
          config.per_layer_projection_shape,
          layer.ple_projection)) {
        error_message = "Gemma token graph PLE data is inconsistent.";
        return false;
      }
      layer.ple_post_norm = &cached_tensor_vector_f32(
        state, layer_index.post_norm, "Gemma per-layer post norm");
    }
    if (layer_index.layer_output_scale != k_missing_tensor_index) {
      const std::vector<float> &scale = cached_tensor_vector_f32(
        state, layer_index.layer_output_scale, "Gemma layer output scale");
      if (scale.size() != 1) {
        error_message = "Gemma token graph layer output scale is not scalar.";
        return false;
      }
      layer.output_scale = scale[0];
    }

    layer.attention.cache_id =
      session.gpu_kv_cache_generation * 1315423911ull
      + static_cast<std::uint64_t>(kv_source_layer + 1);
    layer.attention.position = static_cast<std::uint32_t>(position);
    layer.attention.max_positions = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
        session.gpu_kv_max_positions,
        std::numeric_limits<std::uint32_t>::max()));
    layer.attention.head_count = static_cast<std::uint32_t>(config.head_count);
    layer.attention.kv_head_count =
      static_cast<std::uint32_t>(config.kv_head_count);
    layer.attention.head_dim = static_cast<std::uint32_t>(config.head_dim);
    layer.attention.attention_window = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(
        config.attention_window,
        std::numeric_limits<std::uint32_t>::max()));
    layer.attention.rms_epsilon = static_cast<float>(config.rms_norm_epsilon);
    layer.attention.rope_freq_base = static_cast<float>(config.rope_freq_base);
    layer.attention.rope_freq_scale = static_cast<float>(config.rope_freq_scale);
    gpu_layers.push_back(std::move(layer));
  }

  {
    const TensorInfo &output_tensor =
      tensor_at(state, state.tensor_index.output);
    const MatrixShape output_shape =
      tensor_matrix_shape(output_tensor, "Gemma token graph LM head");
    if (!build_resident_ref(
      state, output_tensor, output_shape, session.gpu_token_graph_output_ref)) {
      error_message = "Gemma token graph LM head format is unsupported.";
      return false;
    }
  }
  session.gpu_token_graph_descriptors_ready = true;
  session.gpu_token_graph_descriptor_generation = session.gpu_kv_cache_generation;
  }

  // Per-token patching of the cached descriptors.
  if (per_layer_input_size > 0) {
    scaled_ple_token_inputs = tensor_row_f32(
      state,
      tensor_at(state, state.tensor_index.per_layer_token_embeddings),
      token_id);
    scale_vector_in_place(
      scaled_ple_token_inputs,
      std::sqrt(static_cast<double>(per_layer_input_size)));
    gpu_ple_inputs.token_inputs = &scaled_ple_token_inputs;
    gpu_ple_inputs_ptr = &gpu_ple_inputs;
  }
  for (D3d12GemmaTokenLayer &cached_layer : gpu_layers) {
    cached_layer.attention.position = static_cast<std::uint32_t>(position);
  }
  std::uint32_t selected_token = 0;
  const bool success = d3d12_gemma_token_resident(
    initial_hidden,
    gpu_layers,
    gpu_ple_inputs_ptr,
    cached_tensor_vector_f32(
      state, state.tensor_index.output_norm, "Gemma output norm"),
    session.gpu_token_graph_output_ref,
    static_cast<float>(state.model.layer_norm_rms_epsilon.value_or(1e-5)),
    static_cast<float>(state.model.final_logit_softcapping.value_or(0.0)),
    &final_hidden,
    need_logits ? &final_logits : nullptr,
    &selected_token,
    &error_message);
  if (!success) {
    return false;
  }
  selected_token_out = selected_token;
  for (std::size_t layer_number = 0;
    layer_number < state.gemma_layer_configs.size();
    layer_number += 1) {
    const std::size_t kv_source_layer = resolve_gemma_kv_source_layer(
      state, layer_number, state.gemma_layer_configs[layer_number]);
    session.gpu_kv_cache_active[kv_source_layer] = true;
  }
  return true;
}

static void execute_gemma_consume_token(
  const LoadedModelState &state,
  SessionState &session,
  std::uint64_t token_id,
  bool compute_logits = true) {
  if (!is_gemma_architecture(state.model.architecture)) {
    throw WorkerError("NATIVE_EXECUTOR_UNIMPLEMENTED", "The loaded GGUF is not a Gemma-family model.");
  }
  if (session.layered_kv_cache.size() != state.tensor_index.layers.size()) {
    session.layered_kv_cache.assign(state.tensor_index.layers.size(), SessionState::LayerKvCache {});
  }

  std::vector<float> hidden_state = tensor_row_f32(
    state,
    tensor_at(state, state.tensor_index.token_embeddings),
    token_id);
  scale_vector_in_place(hidden_state, std::sqrt(static_cast<double>(state.model.embedding_length.value_or(1))));
  const std::uint64_t position = session.cached_token_count;
  session.last_gpu_argmax_token = -1;
  if (gpu_decode_lane_enabled_for_model(state)
    && gpu_gemma_attention_graph_enabled()
    && gpu_gemma_token_graph_enabled()) {
    const auto graph_started_at = std::chrono::steady_clock::now();
    const std::uint64_t graph_gpu_before = g_decode_gpu_use_count;
    const std::string graph_detail_before = g_decode_gpu_detail;
    // Greedy fast path: the GPU argmax replaces the logits readback.
    const bool greedy_fastpath = compute_logits
      && session.sampling.temperature <= 0.0
      && gpu_greedy_fastpath_enabled();
    std::vector<float> graph_hidden;
    std::vector<float> graph_logits;
    std::uint32_t graph_selected_token = 0;
    std::string graph_error;
    if (try_execute_gemma_token_graph(
      state,
      session,
      token_id,
      hidden_state,
      graph_hidden,
      graph_logits,
      compute_logits && !greedy_fastpath,
      graph_selected_token,
      graph_error)) {
      mark_decode_gpu_used(
        "Direct3D 12 complete resident Gemma token graph executed on "
        + d3d12_batch_matmul_capabilities().adapter_name + ".");
      record_decode_phase_metric(
        session,
        "Gemma complete resident token graph",
        -1,
        graph_started_at,
        graph_gpu_before,
        graph_detail_before);
      session.sequence_token_ids.push_back(token_id);
      session.cached_token_count += 1;
      if (greedy_fastpath) {
        session.last_hidden_state = std::move(graph_hidden);
        session.last_logits.clear();
        session.last_gpu_argmax_token =
          static_cast<std::int64_t>(graph_selected_token);
      } else if (compute_logits) {
        session.last_hidden_state = std::move(graph_hidden);
        session.last_logits = std::move(graph_logits);
      } else {
        session.last_hidden_state.clear();
        session.last_logits.clear();
      }
      return;
    }
    const bool active_gpu_cache = std::any_of(
      session.gpu_kv_cache_active.begin(),
      session.gpu_kv_cache_active.end(),
      [](bool active) { return active; });
    if (active_gpu_cache || g_gpu_decode_required) {
      throw_required_gpu_decode_failure("Gemma complete token graph", graph_error);
    }
  }
  const std::vector<float> per_layer_inputs = build_gemma_per_layer_inputs(
    state,
    token_id,
    hidden_state);

  for (std::size_t layer_number = 0; layer_number < state.tensor_index.layers.size(); layer_number += 1) {
    const LlamaLayerTensorIndex &layer_index = state.tensor_index.layers[layer_number];
    std::optional<GemmaLayerExecutionConfig> resolved_config;
    const GemmaLayerExecutionConfig *config_ptr = nullptr;
    if (state.gemma_layer_configs_ready && layer_number < state.gemma_layer_configs.size()) {
      config_ptr = &state.gemma_layer_configs[layer_number];
    } else {
      resolved_config = resolve_gemma_layer_execution_config(state, layer_index);
      if (!resolved_config.has_value()) {
        throw WorkerError("NATIVE_EXECUTOR_UNIMPLEMENTED", "The loaded GGUF layer is outside the native Gemma execution subset.");
      }
      config_ptr = &resolved_config.value();
    }
    const GemmaLayerExecutionConfig &config = *config_ptr;
    const std::size_t kv_source_layer = resolve_gemma_kv_source_layer(
      state,
      layer_number,
      config);
    SessionState::LayerKvCache &layer_cache = session.layered_kv_cache[kv_source_layer];
    // Holds the resident GPU layer output in verify mode so it can be diffed
    // against the CPU recomputation at the end of this layer iteration.
    std::optional<std::vector<float>> verify_gpu_hidden;
    // Attention context computed by the standalone resident attention kernel
    // in verify mode (separate cache namespace; compared against CPU attention).
    std::optional<std::vector<float>> verify_gpu_context;
    const bool owns_kv_cache = kv_source_layer == layer_number;
    if (kv_source_layer >= state.gemma_layer_configs.size()) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Gemma shared-KV source layer is outside the execution configuration.");
    }
    const GemmaLayerExecutionConfig &kv_config = state.gemma_layer_configs[kv_source_layer];
    if (kv_config.kv_size != config.kv_size || kv_config.head_dim != config.head_dim) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Gemma shared-KV source layer has incompatible attention dimensions.");
    }

    const std::vector<float> &attention_norm_weights = cached_tensor_vector_f32(state, layer_index.attention_norm, "Gemma attention norm");
    const std::vector<float> &query_norm_weights = cached_tensor_vector_f32(state, layer_index.query_norm, "Gemma query norm");
    const std::vector<float> &key_norm_weights = cached_tensor_vector_f32(state, layer_index.key_norm, "Gemma key norm");
    const std::vector<float> &post_attention_norm_weights = cached_tensor_vector_f32(state, layer_index.post_attention_norm, "Gemma post-attention norm");
    const std::vector<float> &ffn_norm_weights = cached_tensor_vector_f32(state, layer_index.ffn_norm, "Gemma FFN norm");
    const std::vector<float> &post_ffw_norm_weights = cached_tensor_vector_f32(state, layer_index.post_ffw_norm, "Gemma post-FFW norm");

    const TensorInfo &query_tensor = tensor_at(state, layer_index.query);
    const TensorInfo &key_tensor = tensor_at(state, layer_index.key);
    const TensorInfo &value_tensor = tensor_at(state, layer_index.value);
    const TensorInfo &output_tensor = tensor_at(state, layer_index.output);
    const TensorInfo &gate_tensor = tensor_at(state, layer_index.gate);
    const TensorInfo &down_tensor = tensor_at(state, layer_index.down);
    const TensorInfo &up_tensor = tensor_at(state, layer_index.up);

    const MatrixShape &query_shape = config.query_shape;
    const MatrixShape &key_shape = config.key_shape;
    const MatrixShape &value_shape = config.value_shape;
    const MatrixShape &output_shape = config.output_shape;
    const MatrixShape &gate_shape = config.gate_shape;
    const MatrixShape &down_shape = config.down_shape;
    const MatrixShape &up_shape = config.up_shape;

    std::vector<float> per_layer_input;
    const std::vector<float> *per_layer_post_norm_weights = nullptr;
    if (config.per_layer_input_size > 0) {
      const std::size_t per_layer_offset =
        layer_number * static_cast<std::size_t>(config.per_layer_input_size);
      if (per_layer_offset + static_cast<std::size_t>(config.per_layer_input_size) > per_layer_inputs.size()) {
        throw WorkerError("NATIVE_WORKER_ERROR", "Gemma per-layer input slice is out of bounds.");
      }
      per_layer_input.assign(
        per_layer_inputs.begin() + static_cast<std::ptrdiff_t>(per_layer_offset),
        per_layer_inputs.begin() + static_cast<std::ptrdiff_t>(
          per_layer_offset + static_cast<std::size_t>(config.per_layer_input_size)));
      per_layer_post_norm_weights = &cached_tensor_vector_f32(
        state,
        layer_index.post_norm,
        "Gemma per-layer post norm");
    }

    const std::vector<float> normed = rms_norm(hidden_state, attention_norm_weights, config.rms_norm_epsilon);
    const std::uint64_t cached_positions = session.cached_token_count + 1;
    std::vector<float> attention_context;
    bool resident_gpu_attention = false;
    bool resident_gpu_layer = false;
    const auto resident_attention_started_at = std::chrono::steady_clock::now();
    const std::uint64_t resident_attention_gpu_before = g_decode_gpu_use_count;
    const std::string resident_attention_detail_before = g_decode_gpu_detail;
    if (gpu_decode_lane_enabled_for_model(state) && gpu_gemma_attention_graph_enabled()) {
      if (session.gpu_kv_cache_generation == 0) {
        static std::atomic<std::uint64_t> next_gpu_kv_cache_generation {1};
        session.gpu_kv_cache_generation =
          next_gpu_kv_cache_generation.fetch_add(1, std::memory_order_relaxed);
      }
      if (session.gpu_kv_cache_active.size() != state.tensor_index.layers.size()) {
        session.gpu_kv_cache_active.assign(state.tensor_index.layers.size(), false);
      }
      D3d12ResidentWeightRef query_ref;
      D3d12ResidentWeightRef key_ref;
      D3d12ResidentWeightRef value_ref;
      D3d12ResidentWeightRef attn_out_ref;
      D3d12ResidentWeightRef gate_ref;
      D3d12ResidentWeightRef up_ref;
      D3d12ResidentWeightRef down_ref;
      D3d12ResidentWeightRef ple_gate_ref;
      D3d12ResidentWeightRef ple_projection_ref;
      const bool ple_ready = config.per_layer_input_size == 0
        || (build_resident_ref(
          state,
          tensor_at(state, layer_index.per_layer_input_gate),
          config.per_layer_input_gate_shape,
          ple_gate_ref)
          && build_resident_ref(
            state,
            tensor_at(state, layer_index.per_layer_projection),
            config.per_layer_projection_shape,
            ple_projection_ref));
      // Resident GPU attention seeds its cache from the f32/f16 KV lanes
      // (uploaded as halves either way); quantized lanes keep this layer on
      // the CPU path until a quantized upload path exists.
      const bool refs_ready = !kv_cache_mode_is_quantized()
        && build_resident_ref(state, query_tensor, query_shape, query_ref)
        && (!owns_kv_cache
          || (build_resident_ref(state, key_tensor, key_shape, key_ref)
            && build_resident_ref(state, value_tensor, value_shape, value_ref)))
        && build_resident_ref(state, output_tensor, output_shape, attn_out_ref)
        && build_resident_ref(state, gate_tensor, gate_shape, gate_ref)
        && build_resident_ref(state, up_tensor, up_shape, up_ref)
        && build_resident_ref(state, down_tensor, down_shape, down_ref)
        && ple_ready;
      if (refs_ready) {
        D3d12GemmaAttentionConfig gpu_config;
        gpu_config.cache_id =
          session.gpu_kv_cache_generation * 1315423911ull
          + static_cast<std::uint64_t>(kv_source_layer + 1);
        gpu_config.position = static_cast<std::uint32_t>(position);
        const std::uint64_t requested_cache_horizon = position
          + std::max<std::uint64_t>(session.max_tokens, 1)
          + 2;
        if (session.gpu_kv_max_positions == 0) {
          session.gpu_kv_max_positions = std::min<std::uint64_t>(
            state.model.context_length.value_or(requested_cache_horizon),
            requested_cache_horizon);
        }
        gpu_config.max_positions = static_cast<std::uint32_t>(std::min<std::uint64_t>(
          session.gpu_kv_max_positions,
          std::numeric_limits<std::uint32_t>::max()));
        gpu_config.head_count = static_cast<std::uint32_t>(config.head_count);
        gpu_config.kv_head_count = static_cast<std::uint32_t>(config.kv_head_count);
        gpu_config.head_dim = static_cast<std::uint32_t>(config.head_dim);
        gpu_config.attention_window = static_cast<std::uint32_t>(std::min<std::uint64_t>(
          config.attention_window,
          std::numeric_limits<std::uint32_t>::max()));
        gpu_config.rms_epsilon = static_cast<float>(config.rms_norm_epsilon);
        gpu_config.rope_freq_base = static_cast<float>(config.rope_freq_base);
        gpu_config.rope_freq_scale = static_cast<float>(config.rope_freq_scale);
        std::string gpu_attention_error;
        std::vector<float> resident_hidden;
        resident_gpu_layer = d3d12_gemma_layer_resident(
          query_ref,
          owns_kv_cache ? &key_ref : nullptr,
          owns_kv_cache ? &value_ref : nullptr,
          attn_out_ref,
          gate_ref,
          up_ref,
          down_ref,
          config.per_layer_input_size > 0 ? &ple_gate_ref : nullptr,
          config.per_layer_input_size > 0 ? &ple_projection_ref : nullptr,
          normed,
          hidden_state,
          query_norm_weights,
          key_norm_weights,
          config.rope_freq_factors,
          layer_cache.keys.gpu_lane_view(),
          layer_cache.values.gpu_lane_view(),
          post_attention_norm_weights,
          ffn_norm_weights,
          post_ffw_norm_weights,
          config.per_layer_input_size > 0 ? &per_layer_input : nullptr,
          per_layer_post_norm_weights,
          gpu_config,
          &resident_hidden,
          &gpu_attention_error);
        resident_gpu_attention = resident_gpu_layer;
        if (resident_gpu_layer && gpu_attn_verify_enabled()) {
          // Verify mode: stash the GPU result, then let the trusted CPU path
          // recompute this layer so the outputs can be compared at layer end.
          verify_gpu_hidden = std::move(resident_hidden);
          session.gpu_kv_cache_active[kv_source_layer] = true;
          resident_gpu_layer = false;
          resident_gpu_attention = false;
          // Also compute the attention context alone through the standalone
          // resident kernel in a private cache namespace for comparison
          // against the CPU attention below.
          // Projection-level check: the batch-one matvec route against the
          // CPU dequantized dot for the SAME query tensor and input.
          {
            std::vector<float> gpu_projection;
            if (d3d12_batch_matmul_resident(
              &query_ref, 1, normed, 1, next_gpu_input_identity(), &gpu_projection, nullptr)) {
              std::vector<float> cpu_projection(query_shape.row_count, 0.0f);
              for (std::uint64_t row = 0; row < query_shape.row_count; row += 1) {
                cpu_projection[static_cast<std::size_t>(row)] = static_cast<float>(tensor_range_dot_f32(
                  state,
                  query_tensor,
                  row * query_shape.row_width,
                  query_shape.row_width,
                  normed,
                  "attn-verify query projection"));
              }
              double proj_max = 0.0;
              std::size_t proj_at = 0;
              for (std::size_t item = 0; item < cpu_projection.size() && item < gpu_projection.size(); item += 1) {
                const double diff = std::abs(
                  static_cast<double>(gpu_projection[item]) - static_cast<double>(cpu_projection[item]));
                if (diff > proj_max) {
                  proj_max = diff;
                  proj_at = item;
                }
              }
              std::fprintf(
                stderr,
                "attn-verify-proj layer=%zu max_abs=%.6f at=%zu gpu_v=%.4f cpu_v=%.4f\n",
                layer_number,
                proj_max,
                proj_at,
                proj_at < gpu_projection.size() ? gpu_projection[proj_at] : 0.0f,
                proj_at < cpu_projection.size() ? cpu_projection[proj_at] : 0.0f);
            }
          }
          D3d12GemmaAttentionConfig verify_config = gpu_config;
          verify_config.cache_id = gpu_config.cache_id + 900000000ull;
          std::vector<float> verify_context;
          std::string verify_error;
          if (d3d12_gemma_attention_resident(
            query_ref,
            owns_kv_cache ? &key_ref : nullptr,
            owns_kv_cache ? &value_ref : nullptr,
            normed,
            query_norm_weights,
            key_norm_weights,
            config.rope_freq_factors,
            layer_cache.keys.gpu_lane_view(),
            layer_cache.values.gpu_lane_view(),
            verify_config,
            &verify_context,
            &verify_error)) {
            verify_gpu_context = std::move(verify_context);
          } else {
            std::fprintf(
              stderr,
              "attn-verify layer=%zu standalone attention failed: %s\n",
              layer_number,
              verify_error.c_str());
          }
        } else if (resident_gpu_layer) {
          hidden_state = std::move(resident_hidden);
          session.gpu_kv_cache_active[kv_source_layer] = true;
          mark_decode_gpu_used(
            "Direct3D 12 resident full Gemma layer graph executed on "
            + d3d12_batch_matmul_capabilities().adapter_name + ".");
        } else if (session.gpu_kv_cache_active[kv_source_layer] || g_gpu_decode_required) {
          throw_required_gpu_decode_failure("Gemma resident attention", gpu_attention_error);
        }
      } else if (g_gpu_decode_required) {
        throw_required_gpu_decode_failure(
          "Gemma resident attention",
          "one or more Q/K/V tensor formats are unsupported");
      }
    }
    if (!resident_gpu_attention) {
    const auto qkv_started_at = std::chrono::steady_clock::now();
    const std::uint64_t qkv_gpu_before = g_decode_gpu_use_count;
    const std::string qkv_detail_before = g_decode_gpu_detail;
    const bool needs_query = cached_positions > 1;
    std::vector<float> query;
    std::vector<float> fused_key;
    std::vector<float> fused_value;
    bool fused_qkv = false;
    if (needs_query && owns_kv_cache && !session.decode_qkv_fuse_blocked) {
      const TensorInfo *qkv_tensors[3] = {&query_tensor, &key_tensor, &value_tensor};
      const MatrixShape *qkv_shapes[3] = {&query_shape, &key_shape, &value_shape};
      std::vector<float> *qkv_outputs[3] = {&query, &fused_key, &fused_value};
      fused_qkv = try_multi_matvec_resident(state, qkv_tensors, qkv_shapes, 3, normed, qkv_outputs);
      if (!fused_qkv) {
        session.decode_qkv_fuse_blocked = true;
        session.decode_qkv_fuse_detail =
          "Direct3D 12 fused decode QKV fell back to split projections for this session.";
      }
    }
    if (needs_query) {
      if (!fused_qkv) {
        query = matrix_vector_mul_tensor(
          state,
          query_tensor,
          query_shape,
          normed,
          "Gemma attention query projection");
      }
      query = rms_norm_heads(
        query,
        query_norm_weights,
        config.head_count,
        config.head_dim,
        "Gemma attention query norm",
        config.rms_norm_epsilon);
      apply_neox_rope_in_place(
        query,
        position,
        config.head_count,
        config.head_dim,
        config.rope_freq_base,
        config.rope_freq_scale,
        config.rope_freq_factors);
    }
    if (owns_kv_cache) {
      std::vector<float> key = fused_qkv
        ? std::move(fused_key)
        : matrix_vector_mul_tensor(
          state,
          key_tensor,
          key_shape,
          normed,
          "Gemma attention key projection");
      std::vector<float> value = fused_qkv
        ? std::move(fused_value)
        : matrix_vector_mul_tensor(
          state,
          value_tensor,
          value_shape,
          normed,
          "Gemma attention value projection");
      value = rms_norm_heads_unweighted(
        value,
        config.kv_head_count,
        config.head_dim,
        "Gemma attention value norm",
        config.rms_norm_epsilon);

      key = rms_norm_heads(
        key,
        key_norm_weights,
        config.kv_head_count,
        config.head_dim,
        "Gemma attention key norm",
        config.rms_norm_epsilon);

      apply_neox_rope_in_place(
        key,
        position,
        config.kv_head_count,
        config.head_dim,
        config.rope_freq_base,
        config.rope_freq_scale,
        config.rope_freq_factors);

      append_layer_cache(layer_cache, key, value);
    }
    record_decode_phase_metric(
      session,
      "Gemma decode attention QKV",
      static_cast<std::int64_t>(layer_number),
      qkv_started_at,
      qkv_gpu_before,
      qkv_detail_before);
    if (layer_cache.keys.size() != static_cast<std::size_t>(cached_positions * config.kv_size)
      || layer_cache.values.size() != static_cast<std::size_t>(cached_positions * config.kv_size)) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Native Gemma KV cache size is inconsistent with the current session state.");
    }

    const auto attention_cpu_started_at = std::chrono::steady_clock::now();
    const std::uint64_t attention_cpu_gpu_before = g_decode_gpu_use_count;
    const std::string attention_cpu_detail_before = g_decode_gpu_detail;
    attention_context.assign(static_cast<std::size_t>(config.query_size), 0.0f);
    const auto compute_attention_heads = [&](std::uint64_t start_head, std::uint64_t end_head) {
      std::vector<float> scores;
      std::vector<float> kv_row_scratch;
      for (std::uint64_t head = start_head; head < end_head; head += 1) {
        const std::uint64_t kv_head = head * config.kv_head_count / config.head_count;
        const std::size_t query_offset = static_cast<std::size_t>(head * config.head_dim);
        float *context_ptr = attention_context.data() + static_cast<std::ptrdiff_t>(query_offset);
        if (cached_positions == 1) {
          const std::size_t source_base = static_cast<std::size_t>(kv_head * config.head_dim);
          const float *value_ptr = layer_cache.values.row_f32(
            source_base, static_cast<std::size_t>(config.head_dim), kv_row_scratch);
          for (std::uint64_t item = 0; item < config.head_dim; item += 1) {
            context_ptr[static_cast<std::size_t>(item)] = value_ptr[static_cast<std::size_t>(item)];
          }
        } else {
          const std::uint64_t first_source = config.attention_window > 0
            && cached_positions > config.attention_window
            ? cached_positions - config.attention_window
            : 0;
          const std::uint64_t attended_positions = cached_positions - first_source;
          const float *query_ptr = query.data() + static_cast<std::ptrdiff_t>(query_offset);
          scores.assign(static_cast<std::size_t>(attended_positions), 0.0f);
          for (std::uint64_t source = first_source; source < cached_positions; source += 1) {
            float dot = 0.0f;
            const std::size_t source_base = static_cast<std::size_t>(source * config.kv_size + kv_head * config.head_dim);
            const float *key_ptr = layer_cache.keys.row_f32(
              source_base, static_cast<std::size_t>(config.head_dim), kv_row_scratch);
            dot += attention_dot(query_ptr, key_ptr, static_cast<std::size_t>(config.head_dim));
            scores[static_cast<std::size_t>(source - first_source)] = dot;
          }
          softmax_in_place(scores);
          for (std::uint64_t source = first_source; source < cached_positions; source += 1) {
            const std::size_t source_base = static_cast<std::size_t>(source * config.kv_size + kv_head * config.head_dim);
            const float weight = scores[static_cast<std::size_t>(source - first_source)];
            const float *value_ptr = layer_cache.values.row_f32(
              source_base, static_cast<std::size_t>(config.head_dim), kv_row_scratch);
            attention_axpy(context_ptr, value_ptr, weight, static_cast<std::size_t>(config.head_dim));
          }
        }
      }
    };
    const std::uint64_t attention_worker_count = choose_row_worker_count(
      config.head_count * cached_positions * config.head_dim,
      config.head_count,
      16384,
      1);
    if (attention_worker_count <= 1) {
      compute_attention_heads(0, config.head_count);
    } else {
      persistent_row_workers().execute(config.head_count, attention_worker_count, compute_attention_heads);
    }
    record_decode_phase_metric(
      session,
      "Gemma decode attention CPU",
      static_cast<std::int64_t>(layer_number),
      attention_cpu_started_at,
      attention_cpu_gpu_before,
      attention_cpu_detail_before);
    if (verify_gpu_context.has_value()) {
      const std::vector<float> &gpu_ctx = verify_gpu_context.value();
      double ctx_max = 0.0;
      double ctx_sum = 0.0;
      std::size_t ctx_at = 0;
      const std::size_t ctx_count = std::min(gpu_ctx.size(), attention_context.size());
      for (std::size_t item = 0; item < ctx_count; item += 1) {
        const double diff = std::abs(
          static_cast<double>(gpu_ctx[item]) - static_cast<double>(attention_context[item]));
        ctx_sum += diff;
        if (diff > ctx_max) {
          ctx_max = diff;
          ctx_at = item;
        }
      }
      std::fprintf(
        stderr,
        "attn-verify-ctx layer=%zu pos=%llu max_abs=%.6f mean_abs=%.6f gpu_n=%zu cpu_n=%zu heads=%llu kv=%llu dim=%llu qsize=%llu window=%llu\n",
        layer_number,
        static_cast<unsigned long long>(position),
        ctx_max,
        ctx_count > 0 ? ctx_sum / static_cast<double>(ctx_count) : 0.0,
        gpu_ctx.size(),
        attention_context.size(),
        static_cast<unsigned long long>(config.head_count),
        static_cast<unsigned long long>(config.kv_head_count),
        static_cast<unsigned long long>(config.head_dim),
        static_cast<unsigned long long>(config.query_size),
        static_cast<unsigned long long>(config.attention_window));
      verify_gpu_context.reset();
    }
    } else {
      record_decode_phase_metric(
        session,
        "Gemma decode resident full layer graph",
        static_cast<std::int64_t>(layer_number),
        resident_attention_started_at,
        resident_attention_gpu_before,
        resident_attention_detail_before);
    }

    // Fused decode layer tail: attention-output matmul, both residual adds,
    // all three RMS-norms, and the FFN block execute as ONE GPU submission,
    // replacing two submissions plus the CPU elementwise work between them.
    const auto fused_tail_started_at = std::chrono::steady_clock::now();
    const std::uint64_t fused_tail_gpu_before = g_decode_gpu_use_count;
    const std::string fused_tail_detail_before = g_decode_gpu_detail;
    bool fused_layer_tail = resident_gpu_layer;
    bool fused_layer_ple = resident_gpu_layer && config.per_layer_input_size > 0;
    if (!resident_gpu_layer
      && gpu_decode_lane_enabled_for_model(state)
      && gpu_fuse_layer_enabled()
      && !session.decode_layer_tail_fuse_blocked) {
      D3d12ResidentWeightRef attn_out_ref;
      D3d12ResidentWeightRef gate_ref;
      D3d12ResidentWeightRef up_ref;
      D3d12ResidentWeightRef down_ref;
      D3d12ResidentWeightRef ple_gate_ref;
      D3d12ResidentWeightRef ple_projection_ref;
      const bool ple_ready = config.per_layer_input_size > 0
        && build_resident_ref(
          state,
          tensor_at(state, layer_index.per_layer_input_gate),
          config.per_layer_input_gate_shape,
          ple_gate_ref)
        && build_resident_ref(
          state,
          tensor_at(state, layer_index.per_layer_projection),
          config.per_layer_projection_shape,
          ple_projection_ref);
      if (build_resident_ref(state, output_tensor, output_shape, attn_out_ref)
        && build_resident_ref(state, gate_tensor, gate_shape, gate_ref)
        && build_resident_ref(state, up_tensor, up_shape, up_ref)
        && build_resident_ref(state, down_tensor, down_shape, down_ref)) {
        std::vector<float> fused_hidden;
        std::string fused_tail_error;
        const bool fused_success = ple_ready
          ? d3d12_gemma_layer_tail_ple_resident(
            attn_out_ref,
            gate_ref,
            up_ref,
            down_ref,
            ple_gate_ref,
            ple_projection_ref,
            attention_context,
            hidden_state,
            post_attention_norm_weights,
            ffn_norm_weights,
            post_ffw_norm_weights,
            per_layer_input,
            *per_layer_post_norm_weights,
            static_cast<float>(config.rms_norm_epsilon),
            &fused_hidden,
            &fused_tail_error)
          : d3d12_gemma_layer_tail_resident(
            attn_out_ref,
            gate_ref,
            up_ref,
            down_ref,
            attention_context,
            hidden_state,
            post_attention_norm_weights,
            ffn_norm_weights,
            post_ffw_norm_weights,
            static_cast<float>(config.rms_norm_epsilon),
            &fused_hidden,
            &fused_tail_error);
        if (fused_success) {
          bool finite = true;
          for (const float value : fused_hidden) {
            if (!std::isfinite(value)) {
              finite = false;
              break;
            }
          }
          if (finite) {
            mark_decode_gpu_used(
              "Direct3D 12 resident fused Gemma decode layer tail executed on "
              + d3d12_batch_matmul_capabilities().adapter_name + ".");
            hidden_state = std::move(fused_hidden);
            fused_layer_tail = true;
            fused_layer_ple = ple_ready;
          } else {
            session.decode_layer_tail_fuse_blocked = true;
            session.decode_layer_tail_fuse_detail =
              "Direct3D 12 fused Gemma decode layer tail produced non-finite values.";
          }
        } else {
          session.decode_layer_tail_fuse_blocked = true;
          session.decode_layer_tail_fuse_detail = fused_tail_error.empty()
            ? "Direct3D 12 fused Gemma decode layer tail fell back to split decode phases."
            : fused_tail_error;
        }
      }
    }
    if (!fused_layer_tail) {
      const auto output_started_at = std::chrono::steady_clock::now();
      const std::uint64_t output_gpu_before = g_decode_gpu_use_count;
      const std::string output_detail_before = g_decode_gpu_detail;
      std::vector<float> attention_output = matrix_vector_mul_tensor(
        state,
        output_tensor,
        output_shape,
        attention_context,
        "Gemma attention output projection");
      record_decode_phase_metric(
        session,
        "Gemma decode attention output",
        static_cast<std::int64_t>(layer_number),
        output_started_at,
        output_gpu_before,
        output_detail_before);
      attention_output = rms_norm(attention_output, post_attention_norm_weights, config.rms_norm_epsilon);
      const std::vector<float> residual = add_vectors(hidden_state, attention_output, "Gemma attention residual");
      const std::vector<float> ffn_normed = rms_norm(residual, ffn_norm_weights, config.rms_norm_epsilon);
      const auto ffn_started_at = std::chrono::steady_clock::now();
      const std::uint64_t ffn_gpu_before = g_decode_gpu_use_count;
      const std::string ffn_detail_before = g_decode_gpu_detail;
      std::vector<float> down;
      bool fused_ffn = false;
      if (!session.decode_ffn_fuse_blocked) {
        fused_ffn = try_ffn_chain_matvec_resident(
          state,
          gate_tensor,
          gate_shape,
          up_tensor,
          up_shape,
          down_tensor,
          down_shape,
          ffn_normed,
          down);
        if (!fused_ffn) {
          session.decode_ffn_fuse_blocked = true;
          session.decode_ffn_fuse_detail =
            "Direct3D 12 fused decode FFN chain fell back to split FFN projections for this session.";
        }
      }
      if (!fused_ffn) {
        std::vector<float> gelu_glu = fused_gelu_glu_projection_tensor(
          state,
          gate_tensor,
          gate_shape,
          up_tensor,
          up_shape,
          ffn_normed,
          "Gemma MLP GELU-GLU projection");
        down = matrix_vector_mul_tensor(
          state,
          down_tensor,
          down_shape,
          gelu_glu,
          "Gemma MLP down projection");
      }
      record_decode_phase_metric(
        session,
        "Gemma decode FFN",
        static_cast<std::int64_t>(layer_number),
        ffn_started_at,
        ffn_gpu_before,
        ffn_detail_before);
      down = rms_norm(down, post_ffw_norm_weights, config.rms_norm_epsilon);
      hidden_state = add_vectors(residual, down, "Gemma MLP residual");
    } else if (!resident_gpu_layer) {
      record_decode_phase_metric(
        session,
        "Gemma decode fused layer tail",
        static_cast<std::int64_t>(layer_number),
        fused_tail_started_at,
        fused_tail_gpu_before,
        fused_tail_detail_before);
    }

    if (config.per_layer_input_size > 0 && !fused_layer_ple) {
      std::vector<float> per_layer_gate = matrix_vector_mul_tensor(
        state,
        tensor_at(state, layer_index.per_layer_input_gate),
        config.per_layer_input_gate_shape,
        hidden_state,
        "Gemma per-layer input gate");
      for (std::size_t item = 0; item < per_layer_gate.size(); item += 1) {
        per_layer_gate[item] = gelu_tanh(per_layer_gate[item]) * per_layer_input[item];
      }
      std::vector<float> per_layer_output = matrix_vector_mul_tensor(
        state,
        tensor_at(state, layer_index.per_layer_projection),
        config.per_layer_projection_shape,
        per_layer_gate,
        "Gemma per-layer projection");
      per_layer_output = rms_norm(
        per_layer_output,
        *per_layer_post_norm_weights,
        config.rms_norm_epsilon);
      hidden_state = add_vectors(hidden_state, per_layer_output, "Gemma per-layer residual");
    }
    if (verify_gpu_hidden.has_value()) {
      const std::vector<float> &gpu = verify_gpu_hidden.value();
      double max_diff = 0.0;
      double max_rel = 0.0;
      std::size_t max_index = 0;
      const std::size_t count = std::min(gpu.size(), hidden_state.size());
      for (std::size_t item = 0; item < count; item += 1) {
        const double diff = std::abs(static_cast<double>(gpu[item]) - static_cast<double>(hidden_state[item]));
        if (diff > max_diff) {
          max_diff = diff;
          max_index = item;
        }
        const double denominator = std::max(1e-3, std::abs(static_cast<double>(hidden_state[item])));
        max_rel = std::max(max_rel, diff / denominator);
      }
      double sum_abs = 0.0;
      std::size_t large_count = 0;
      for (std::size_t item = 0; item < count; item += 1) {
        const double diff = std::abs(static_cast<double>(gpu[item]) - static_cast<double>(hidden_state[item]));
        sum_abs += diff;
        if (diff > 0.05) {
          large_count += 1;
        }
      }
      std::fprintf(
        stderr,
        "attn-verify layer=%zu pos=%llu max_abs=%.6f mean_abs=%.6f large=%zu/%zu at=%zu gpu_v=%.4f cpu_v=%.4f\n",
        layer_number,
        static_cast<unsigned long long>(position),
        max_diff,
        count > 0 ? sum_abs / static_cast<double>(count) : 0.0,
        large_count,
        count,
        max_index,
        max_index < gpu.size() ? gpu[max_index] : 0.0f,
        max_index < hidden_state.size() ? hidden_state[max_index] : 0.0f);
      verify_gpu_hidden.reset();
    }
    if (layer_index.layer_output_scale != k_missing_tensor_index) {
      const std::vector<float> &output_scale = cached_tensor_vector_f32(
        state,
        layer_index.layer_output_scale,
        "Gemma layer output scale");
      if (output_scale.size() != 1) {
        throw WorkerError("NATIVE_WORKER_ERROR", "Gemma layer output scale must contain one value.");
      }
      scale_vector_in_place(hidden_state, output_scale[0]);
    }
  }

  session.sequence_token_ids.push_back(token_id);
  session.cached_token_count += 1;
  if (!compute_logits) {
    session.last_hidden_state.clear();
    session.last_logits.clear();
    return;
  }
  session.last_hidden_state = rms_norm(
    hidden_state,
    cached_tensor_vector_f32(state, state.tensor_index.output_norm, "Gemma output norm"),
    state.model.layer_norm_rms_epsilon.value_or(1e-5));
  session.last_logits = project_hidden_to_logits(
    state,
    tensor_at(state, state.tensor_index.output),
    session.last_hidden_state);
  apply_logit_softcap_in_place(
    session.last_logits,
    state.model.final_logit_softcapping.value_or(0.0));
}

// Phase 1 of .agent/vision/01_native_vision_plan.md: prompt positions whose
// hidden state is supplied by the caller instead of looked up from
// token_embd.weight.
//
// This is the whole reason a vision turn can exist. An image becomes a run of
// embedding rows out of the vision tower, and those rows have no token ids -
// there is no integer that means "this patch of this picture". So the prompt
// stops being a list of ids and becomes a list of positions, most of which
// still resolve through the embedding table and some of which arrive already
// computed.
//
// Deliberately position-keyed rather than a flag on the token: the caller knows
// where its image sat in the prompt, and matching on a sentinel token id would
// break the moment a model reuses that id for something else.
//
// Declared here, ahead of BOTH prefill paths - Gemma and llama each need it,
// and Gemma's is defined first.
struct PromptEmbeddingOverrides {
  // position -> hidden state for that position. Rows must be embedding-width.
  std::unordered_map<std::size_t, std::vector<float>> rows;

  const std::vector<float> *row_for(std::size_t position) const {
    const auto found = rows.find(position);
    return found == rows.end() ? nullptr : &found->second;
  }
};

static void execute_gemma_prefill_batch(
  const LoadedModelState &state,
  SessionState &session,
  const std::vector<std::uint64_t> &token_ids,
  BatchExecutionConfig *execution_config,
  FloatBatch *all_position_logits = nullptr,
  const PromptEmbeddingOverrides *embedding_overrides = nullptr) {
  if (token_ids.empty()) {
    return;
  }
  if (!state.gemma_layer_configs_ready
    || state.gemma_layer_configs.size() != state.tensor_index.layers.size()) {
    throw WorkerError("NATIVE_EXECUTOR_UNIMPLEMENTED", "Gemma batched prefill requires validated layer configurations.");
  }
  if (session.layered_kv_cache.size() != state.tensor_index.layers.size()) {
    session.layered_kv_cache.assign(state.tensor_index.layers.size(), SessionState::LayerKvCache {});
  }

  // A supplied row is the FINAL hidden state for its position and is used as
  // given - no embedding scale. Gemma multiplies looked-up token embeddings by
  // sqrt(embedding_length); a projector output is already in the language
  // model's space and scaling it again would quietly wreck it. The override is
  // "here is the hidden state", not "here is a token embedding".
  const TensorInfo &gemma_embedding_table = tensor_at(state, state.tensor_index.token_embeddings);
  FloatBatch hidden_states;
  hidden_states.reserve(token_ids.size());
  for (std::size_t position = 0; position < token_ids.size(); position += 1) {
    const std::vector<float> *supplied = embedding_overrides == nullptr
      ? nullptr
      : embedding_overrides->row_for(position);
    if (supplied != nullptr) {
      const std::size_t expected_width = gemma_embedding_table.dimensions.empty()
        ? supplied->size()
        : static_cast<std::size_t>(gemma_embedding_table.dimensions[0]);
      if (supplied->size() != expected_width) {
        throw WorkerError(
          "NATIVE_EMBEDDING_WIDTH_MISMATCH",
          "A supplied prompt embedding at position " + std::to_string(position)
            + " is " + std::to_string(supplied->size()) + " wide; this model's embedding width is "
            + std::to_string(expected_width) + ".");
      }
      hidden_states.push_back(*supplied);
      continue;
    }
    std::vector<float> hidden_state = tensor_row_f32(state, gemma_embedding_table, token_ids[position]);
    scale_vector_in_place(
      hidden_state,
      std::sqrt(static_cast<double>(state.model.embedding_length.value_or(1))));
    hidden_states.push_back(std::move(hidden_state));
  }
  const FloatBatch per_layer_inputs = build_gemma_per_layer_inputs_batch(
    state,
    token_ids,
    hidden_states,
    execution_config);
  const std::uint64_t initial_cached_positions = session.cached_token_count;

  for (std::size_t layer_number = 0; layer_number < state.tensor_index.layers.size(); layer_number += 1) {
    if (execution_config != nullptr) {
      execution_config->current_layer_index = static_cast<std::int64_t>(layer_number);
    }
    const LlamaLayerTensorIndex &layer_index = state.tensor_index.layers[layer_number];
    const GemmaLayerExecutionConfig &config = state.gemma_layer_configs[layer_number];
    const std::size_t kv_source_layer = resolve_gemma_kv_source_layer(
      state,
      layer_number,
      config);
    if (kv_source_layer >= state.gemma_layer_configs.size()) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Gemma batched shared-KV source layer is outside the execution configuration.");
    }
    const GemmaLayerExecutionConfig &kv_config = state.gemma_layer_configs[kv_source_layer];
    if (kv_config.kv_size != config.kv_size || kv_config.head_dim != config.head_dim) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Gemma batched shared-KV source layer has incompatible attention dimensions.");
    }
    SessionState::LayerKvCache &layer_cache = session.layered_kv_cache[kv_source_layer];
    const bool owns_kv_cache = kv_source_layer == layer_number;

    const std::vector<float> &attention_norm_weights = cached_tensor_vector_f32(
      state,
      layer_index.attention_norm,
      "Gemma attention norm");
    const std::vector<float> &query_norm_weights = cached_tensor_vector_f32(
      state,
      layer_index.query_norm,
      "Gemma query norm");
    const std::vector<float> &key_norm_weights = cached_tensor_vector_f32(
      state,
      layer_index.key_norm,
      "Gemma key norm");
    const std::vector<float> &post_attention_norm_weights = cached_tensor_vector_f32(
      state,
      layer_index.post_attention_norm,
      "Gemma post-attention norm");
    const std::vector<float> &ffn_norm_weights = cached_tensor_vector_f32(
      state,
      layer_index.ffn_norm,
      "Gemma FFN norm");
    const std::vector<float> &post_ffw_norm_weights = cached_tensor_vector_f32(
      state,
      layer_index.post_ffw_norm,
      "Gemma post-FFW norm");

    FloatBatch normed;
    normed.reserve(hidden_states.size());
    for (const std::vector<float> &hidden_state : hidden_states) {
      normed.push_back(rms_norm(
        hidden_state,
        attention_norm_weights,
        config.rms_norm_epsilon));
    }

    FloatBatch queries;
    FloatBatch fused_keys;
    FloatBatch fused_values;
    bool fused_qkv = false;
    if (owns_kv_cache) {
      fused_qkv = try_fused_qkv_batch_resident(
        state,
        tensor_at(state, layer_index.query),
        config.query_shape,
        tensor_at(state, layer_index.key),
        config.key_shape,
        tensor_at(state, layer_index.value),
        config.value_shape,
        normed,
        execution_config,
        queries,
        fused_keys,
        fused_values);
    }
    if (!fused_qkv) {
      queries = matrix_batch_mul_tensor(
        state,
        tensor_at(state, layer_index.query),
        config.query_shape,
        normed,
        execution_config,
        "Gemma batched attention query projection");
    }
    for (std::size_t batch_index = 0; batch_index < queries.size(); batch_index += 1) {
      queries[batch_index] = rms_norm_heads(
        queries[batch_index],
        query_norm_weights,
        config.head_count,
        config.head_dim,
        "Gemma batched attention query norm",
        config.rms_norm_epsilon);
      apply_neox_rope_in_place(
        queries[batch_index],
        initial_cached_positions + static_cast<std::uint64_t>(batch_index),
        config.head_count,
        config.head_dim,
        config.rope_freq_base,
        config.rope_freq_scale,
        config.rope_freq_factors);
    }

    if (owns_kv_cache) {
      FloatBatch keys = fused_qkv
        ? std::move(fused_keys)
        : matrix_batch_mul_tensor(
          state,
          tensor_at(state, layer_index.key),
          config.key_shape,
          normed,
          execution_config,
          "Gemma batched attention key projection");
      FloatBatch values = fused_qkv
        ? std::move(fused_values)
        : matrix_batch_mul_tensor(
          state,
          tensor_at(state, layer_index.value),
          config.value_shape,
          normed,
          execution_config,
          "Gemma batched attention value projection");
      for (std::size_t batch_index = 0; batch_index < keys.size(); batch_index += 1) {
        keys[batch_index] = rms_norm_heads(
          keys[batch_index],
          key_norm_weights,
          config.kv_head_count,
          config.head_dim,
          "Gemma batched attention key norm",
          config.rms_norm_epsilon);
        values[batch_index] = rms_norm_heads_unweighted(
          values[batch_index],
          config.kv_head_count,
          config.head_dim,
          "Gemma batched attention value norm",
          config.rms_norm_epsilon);
        apply_neox_rope_in_place(
          keys[batch_index],
          initial_cached_positions + static_cast<std::uint64_t>(batch_index),
          config.kv_head_count,
          config.head_dim,
          config.rope_freq_base,
          config.rope_freq_scale,
          config.rope_freq_factors);
        append_layer_cache(layer_cache, keys[batch_index], values[batch_index]);
      }
    }

    const std::uint64_t final_cached_positions =
      initial_cached_positions + static_cast<std::uint64_t>(token_ids.size());
    if (layer_cache.keys.size() != static_cast<std::size_t>(final_cached_positions * config.kv_size)
      || layer_cache.values.size() != static_cast<std::size_t>(final_cached_positions * config.kv_size)) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Native batched Gemma KV cache size is inconsistent with the current session state.");
    }

    FloatBatch attention_context(
      hidden_states.size(),
      std::vector<float>(static_cast<std::size_t>(config.query_size), 0.0f));
    for (std::size_t batch_index = 0; batch_index < hidden_states.size(); batch_index += 1) {
      const std::uint64_t cached_positions =
        initial_cached_positions + static_cast<std::uint64_t>(batch_index) + 1;
      const std::uint64_t first_source = config.attention_window > 0
        && cached_positions > config.attention_window
        ? cached_positions - config.attention_window
        : 0;
      const std::uint64_t attended_positions = cached_positions - first_source;
      std::vector<float> kv_row_scratch;
      for (std::uint64_t head = 0; head < config.head_count; head += 1) {
        const std::uint64_t kv_head = head * config.kv_head_count / config.head_count;
        const std::size_t query_offset = static_cast<std::size_t>(head * config.head_dim);
        std::vector<float> scores(static_cast<std::size_t>(attended_positions), 0.0f);
        for (std::uint64_t source = first_source; source < cached_positions; source += 1) {
          double dot = 0.0;
          const std::size_t source_base = static_cast<std::size_t>(
            source * config.kv_size + kv_head * config.head_dim);
          const float *key_ptr = layer_cache.keys.row_f32(
            source_base, static_cast<std::size_t>(config.head_dim), kv_row_scratch);
          for (std::uint64_t item = 0; item < config.head_dim; item += 1) {
            dot += static_cast<double>(
              queries[batch_index][query_offset + static_cast<std::size_t>(item)])
              * static_cast<double>(key_ptr[static_cast<std::size_t>(item)]);
          }
          scores[static_cast<std::size_t>(source - first_source)] = static_cast<float>(dot);
        }
        const std::vector<float> weights = softmax(scores);
        for (std::uint64_t source = first_source; source < cached_positions; source += 1) {
          const std::size_t source_base = static_cast<std::size_t>(
            source * config.kv_size + kv_head * config.head_dim);
          const float *value_ptr = layer_cache.values.row_f32(
            source_base, static_cast<std::size_t>(config.head_dim), kv_row_scratch);
          for (std::uint64_t item = 0; item < config.head_dim; item += 1) {
            attention_context[batch_index][query_offset + static_cast<std::size_t>(item)] +=
              weights[static_cast<std::size_t>(source - first_source)]
              * value_ptr[static_cast<std::size_t>(item)];
      }
    }
  }
  if (execution_config != nullptr) {
    execution_config->current_layer_index = -1;
  }
}

    FloatBatch attention_outputs = matrix_batch_mul_tensor(
      state,
      tensor_at(state, layer_index.output),
      config.output_shape,
      attention_context,
      execution_config,
      "Gemma batched attention output projection");
    FloatBatch residuals;
    residuals.reserve(hidden_states.size());
    FloatBatch ffn_normed;
    ffn_normed.reserve(hidden_states.size());
    for (std::size_t batch_index = 0; batch_index < hidden_states.size(); batch_index += 1) {
      attention_outputs[batch_index] = rms_norm(
        attention_outputs[batch_index],
        post_attention_norm_weights,
        config.rms_norm_epsilon);
      residuals.push_back(add_vectors(
        hidden_states[batch_index],
        attention_outputs[batch_index],
        "Gemma batched attention residual"));
      ffn_normed.push_back(rms_norm(
        residuals.back(),
        ffn_norm_weights,
        config.rms_norm_epsilon));
    }

    FloatBatch down;
    if (!try_ffn_chain_batch_resident(
      state,
      tensor_at(state, layer_index.gate),
      config.gate_shape,
      tensor_at(state, layer_index.up),
      config.up_shape,
      tensor_at(state, layer_index.down),
      config.down_shape,
      ffn_normed,
      execution_config,
      down)) {
      FloatBatch gelu_glu = fused_gelu_glu_projection_tensor_batch(
        state,
        tensor_at(state, layer_index.gate),
        config.gate_shape,
        tensor_at(state, layer_index.up),
        config.up_shape,
        ffn_normed,
        execution_config,
        "Gemma batched MLP GELU-GLU projection");
      down = matrix_batch_mul_tensor(
        state,
        tensor_at(state, layer_index.down),
        config.down_shape,
        gelu_glu,
        execution_config,
        "Gemma batched MLP down projection");
    }
    for (std::size_t batch_index = 0; batch_index < hidden_states.size(); batch_index += 1) {
      down[batch_index] = rms_norm(
        down[batch_index],
        post_ffw_norm_weights,
        config.rms_norm_epsilon);
      hidden_states[batch_index] = add_vectors(
        residuals[batch_index],
        down[batch_index],
        "Gemma batched MLP residual");
    }

    if (config.per_layer_input_size > 0) {
      const std::size_t per_layer_offset =
        layer_number * static_cast<std::size_t>(config.per_layer_input_size);
      FloatBatch per_layer_gate = matrix_batch_mul_tensor(
        state,
        tensor_at(state, layer_index.per_layer_input_gate),
        config.per_layer_input_gate_shape,
        hidden_states,
        execution_config,
        "Gemma batched per-layer input gate");
      for (std::size_t batch_index = 0; batch_index < hidden_states.size(); batch_index += 1) {
        if (per_layer_offset + per_layer_gate[batch_index].size()
          > per_layer_inputs[batch_index].size()) {
          throw WorkerError("NATIVE_WORKER_ERROR", "Gemma batched per-layer input slice is out of bounds.");
        }
        for (std::size_t item = 0; item < per_layer_gate[batch_index].size(); item += 1) {
          per_layer_gate[batch_index][item] = gelu_tanh(per_layer_gate[batch_index][item])
            * per_layer_inputs[batch_index][per_layer_offset + item];
        }
      }
      FloatBatch per_layer_output = matrix_batch_mul_tensor(
        state,
        tensor_at(state, layer_index.per_layer_projection),
        config.per_layer_projection_shape,
        per_layer_gate,
        execution_config,
        "Gemma batched per-layer projection");
      const std::vector<float> &post_norm_weights = cached_tensor_vector_f32(
        state,
        layer_index.post_norm,
        "Gemma per-layer post norm");
      for (std::size_t batch_index = 0; batch_index < hidden_states.size(); batch_index += 1) {
        per_layer_output[batch_index] = rms_norm(
          per_layer_output[batch_index],
          post_norm_weights,
          config.rms_norm_epsilon);
        hidden_states[batch_index] = add_vectors(
          hidden_states[batch_index],
          per_layer_output[batch_index],
          "Gemma batched per-layer residual");
      }
    }
    if (layer_index.layer_output_scale != k_missing_tensor_index) {
      const std::vector<float> &output_scale = cached_tensor_vector_f32(
        state,
        layer_index.layer_output_scale,
        "Gemma layer output scale");
      if (output_scale.size() != 1) {
        throw WorkerError("NATIVE_WORKER_ERROR", "Gemma layer output scale must contain one value.");
      }
      for (std::vector<float> &hidden_state : hidden_states) {
        scale_vector_in_place(hidden_state, output_scale[0]);
      }
    }
  }

  session.sequence_token_ids.insert(
    session.sequence_token_ids.end(),
    token_ids.begin(),
    token_ids.end());
  session.cached_token_count += static_cast<std::uint64_t>(token_ids.size());
  session.last_hidden_state = rms_norm(
    hidden_states.back(),
    cached_tensor_vector_f32(state, state.tensor_index.output_norm, "Gemma output norm"),
    state.model.layer_norm_rms_epsilon.value_or(1e-5));
  session.last_logits = project_hidden_to_logits(
    state,
    tensor_at(state, state.tensor_index.output),
    session.last_hidden_state);
  apply_logit_softcap_in_place(
    session.last_logits,
    state.model.final_logit_softcapping.value_or(0.0));
  // Speculative verification needs the next-token distribution after every
  // fed position, not just the last one. The vocab projection per position is
  // cheap next to the layer stack the batch already paid for.
  if (all_position_logits != nullptr) {
    all_position_logits->clear();
    all_position_logits->reserve(hidden_states.size());
    for (std::size_t index = 0; index + 1 < hidden_states.size(); index += 1) {
      const std::vector<float> normed = rms_norm(
        hidden_states[index],
        cached_tensor_vector_f32(state, state.tensor_index.output_norm, "Gemma output norm"),
        state.model.layer_norm_rms_epsilon.value_or(1e-5));
      std::vector<float> logits = project_hidden_to_logits(
        state,
        tensor_at(state, state.tensor_index.output),
        normed);
      apply_logit_softcap_in_place(
        logits,
        state.model.final_logit_softcapping.value_or(0.0));
      all_position_logits->push_back(std::move(logits));
    }
    all_position_logits->push_back(session.last_logits);
  }
}

// Layer-major batched prefill for plain-llama models, mirroring
// execute_gemma_prefill_batch minus the Gemma-only stages (no embedding
// scaling, no QK/value norms, NEOX-free adjacent-pair RoPE, scaled
// attention scores, no post-attention/post-FFW norms, no PLE, SiLU FFN).
// Until Phase 3 iteration 3 llama prompts ran token-by-token through
// execute_llama_consume_token — one full decode step per prompt token.
// EnhancementRun_04 Phase 4 -- batched decode across SESSIONS.
//
// This is the campaign's load-bearing change, and the reason it is tractable is
// visible in `execute_llama_prefill_batch` below: that function already reads
// each layer's weights ONCE and pushes many rows through them. It just happens
// to fill those rows with successive positions of a single sequence.
//
// Decode is memory-bound -- 2.0115 GB of weights read per emitted token on
// llama-3.2-3b -- so a weight read amortised over B sequences is very nearly a
// B-fold throughput gain until the arithmetic catches up. Measured Phase 0
// starting point: lizard is flat at ~4 tok/s from B=1 to B=4 while llama-server
// climbs 5.58 -> 9.87. That flatness is what this function exists to remove.
//
// Only four things differ from the single-session batch, and all of them are
// per-row bookkeeping rather than kernel work:
//
//   * the RoPE position is each session's own `cached_token_count`, not the
//     row's index within the batch;
//   * KV is appended to each session's own layer cache;
//   * attention runs per session over that session's cache -- deliberately NOT
//     merged across sessions, because each reads different KV so there is no
//     shared read to amortise, and ragged lengths make a merged kernel worse.
//     Attention was measured at 2.4% of decode against matmul's 96%;
//   * every row produces logits, not just the last.
//
// The three batched matmuls -- QKV, attention output, FFN chain -- are called
// exactly as the single-session path calls them, which is what makes B=1
// byte-identity a property of the code rather than a hope.
static void execute_llama_decode_batch(
  const LoadedModelState &state,
  const std::vector<SessionState *> &sessions,
  const std::vector<std::uint64_t> &token_ids,
  BatchExecutionConfig *execution_config,
  FloatBatch *row_logits);

static void execute_llama_prefill_batch(
  const LoadedModelState &state,
  SessionState &session,
  const std::vector<std::uint64_t> &token_ids,
  BatchExecutionConfig *execution_config,
  FloatBatch *all_position_logits = nullptr,
  const PromptEmbeddingOverrides *embedding_overrides = nullptr) {
  if (token_ids.empty()) {
    return;
  }
  const LlamaExecutionConfig *config = cached_llama_execution_config(state);
  if (config == nullptr) {
    throw WorkerError("NATIVE_EXECUTOR_UNIMPLEMENTED", "The loaded GGUF is missing the attention metadata required for native Llama execution.");
  }
  if (!state.llama_layer_shapes_ready
    || state.llama_layer_shapes.size() != state.tensor_index.layers.size()) {
    throw WorkerError("NATIVE_EXECUTOR_UNIMPLEMENTED", "Llama batched prefill requires validated layer shape configurations.");
  }
  if (session.layered_kv_cache.size() != state.tensor_index.layers.size()) {
    session.layered_kv_cache.assign(state.tensor_index.layers.size(), SessionState::LayerKvCache {});
  }

  // Positions resolve through the embedding table unless the caller supplied a
  // hidden state for them. Everything downstream of this loop - RoPE, KV cache,
  // attention, the whole layer stack - is unchanged and cannot tell the
  // difference, which is exactly the property that makes an image splice safe:
  // a supplied row is just a row.
  const TensorInfo &embedding_table = tensor_at(state, state.tensor_index.token_embeddings);
  FloatBatch hidden_states;
  hidden_states.reserve(token_ids.size());
  for (std::size_t position = 0; position < token_ids.size(); position += 1) {
    const std::vector<float> *supplied = embedding_overrides == nullptr
      ? nullptr
      : embedding_overrides->row_for(position);
    if (supplied == nullptr) {
      hidden_states.push_back(tensor_row_f32(state, embedding_table, token_ids[position]));
      continue;
    }
    // A wrong-width row would otherwise read past the end of the batch buffers
    // deep inside the layer loop, where the cause is unrecoverable. Refuse here,
    // where the message can still name the position.
    const std::size_t expected_width = embedding_table.dimensions.empty()
      ? supplied->size()
      : static_cast<std::size_t>(embedding_table.dimensions[0]);
    if (supplied->size() != expected_width) {
      throw WorkerError(
        "NATIVE_EMBEDDING_WIDTH_MISMATCH",
        "A supplied prompt embedding at position " + std::to_string(position)
          + " is " + std::to_string(supplied->size()) + " wide; this model's embedding width is "
          + std::to_string(expected_width) + ".");
    }
    hidden_states.push_back(*supplied);
  }
  const std::uint64_t initial_cached_positions = session.cached_token_count;
  const std::uint64_t kv_hidden_size = config->kv_head_count * config->head_dim;
  const float attention_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(config->head_dim)));

  // Declared outside the layer loop so the storage is allocated once for the
  // whole prefill rather than 28 times. The KV buffers are only populated when
  // the cache is not already f32; in f32 mode row_f32 hands back a direct
  // pointer and these stay empty.
  std::vector<float> keys_dequant_scratch;
  std::vector<float> values_dequant_scratch;
  std::vector<float> keys_head_major;
  std::vector<float> values_head_major;

  for (std::size_t layer_number = 0; layer_number < state.tensor_index.layers.size(); layer_number += 1) {
    if (execution_config != nullptr) {
      execution_config->current_layer_index = static_cast<std::int64_t>(layer_number);
    }
    const LlamaLayerTensorIndex &layer_index = state.tensor_index.layers[layer_number];
    const LlamaLayerShapeConfig &shapes = state.llama_layer_shapes[layer_number];
    SessionState::LayerKvCache &layer_cache = session.layered_kv_cache[layer_number];

    const std::vector<float> &attention_norm_weights = cached_tensor_vector_f32(
      state,
      layer_index.attention_norm,
      "Llama attention norm weights");
    const std::vector<float> &ffn_norm_weights = cached_tensor_vector_f32(
      state,
      layer_index.ffn_norm,
      "Llama FFN norm weights");

    // Prefill stage timings. The projection GEMM was tuned twice on the
    // assumption that it dominates; measured end to end, prefill runs at ~56
    // GFLOP/s while the same kernels hit 310 in isolation, so most of the time
    // is somewhere else. These five labels split a layer into the stages that
    // can actually be acted on, and land in `decodePhaseMetrics` on the prefill
    // response.
    const auto prefill_stage_gpu_count = g_decode_gpu_use_count;
    const std::string prefill_stage_gpu_detail = g_decode_gpu_detail;
    const auto norm_stage_started = std::chrono::steady_clock::now();
    FloatBatch normed;
    normed.reserve(hidden_states.size());
    for (const std::vector<float> &hidden_state : hidden_states) {
      normed.push_back(rms_norm(hidden_state, attention_norm_weights));
    }
    record_decode_phase_metric(
      session, "prefill-attn-norm", static_cast<std::int64_t>(layer_number),
      norm_stage_started, prefill_stage_gpu_count, prefill_stage_gpu_detail);

    const auto qkv_stage_started = std::chrono::steady_clock::now();
    FloatBatch queries;
    FloatBatch keys;
    FloatBatch values;
    if (!try_fused_qkv_batch_resident(
      state,
      tensor_at(state, layer_index.query),
      shapes.query_shape,
      tensor_at(state, layer_index.key),
      shapes.key_shape,
      tensor_at(state, layer_index.value),
      shapes.value_shape,
      normed,
      execution_config,
      queries,
      keys,
      values)) {
      queries = matrix_batch_mul_tensor(
        state,
        tensor_at(state, layer_index.query),
        shapes.query_shape,
        normed,
        execution_config,
        "Llama batched attention query projection");
      keys = matrix_batch_mul_tensor(
        state,
        tensor_at(state, layer_index.key),
        shapes.key_shape,
        normed,
        execution_config,
        "Llama batched attention key projection");
      values = matrix_batch_mul_tensor(
        state,
        tensor_at(state, layer_index.value),
        shapes.value_shape,
        normed,
        execution_config,
        "Llama batched attention value projection");
    }
    record_decode_phase_metric(
      session, "prefill-qkv-projection", static_cast<std::int64_t>(layer_number),
      qkv_stage_started, prefill_stage_gpu_count, prefill_stage_gpu_detail);

    const auto rope_stage_started = std::chrono::steady_clock::now();
    for (std::size_t batch_index = 0; batch_index < queries.size(); batch_index += 1) {
      const std::uint64_t position = initial_cached_positions + static_cast<std::uint64_t>(batch_index);
      apply_rope_in_place(
        queries[batch_index],
        position,
        config->head_count,
        config->head_dim,
        config->rope_freq_base,
        config->rope_freq_scale,
        config->rope_freq_factors);
      apply_rope_in_place(
        keys[batch_index],
        position,
        config->kv_head_count,
        config->head_dim,
        config->rope_freq_base,
        config->rope_freq_scale,
        config->rope_freq_factors);
      append_layer_cache(layer_cache, keys[batch_index], values[batch_index]);
    }

    const std::uint64_t final_cached_positions =
      initial_cached_positions + static_cast<std::uint64_t>(token_ids.size());
    if (layer_cache.keys.size() != static_cast<std::size_t>(final_cached_positions * kv_hidden_size)
      || layer_cache.values.size() != static_cast<std::size_t>(final_cached_positions * kv_hidden_size)) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Native batched Llama KV cache size is inconsistent with the current session state.");
    }

    record_decode_phase_metric(
      session, "prefill-rope-kv-append", static_cast<std::int64_t>(layer_number),
      rope_stage_started, prefill_stage_gpu_count, prefill_stage_gpu_detail);

    const auto attention_stage_started = std::chrono::steady_clock::now();
    FloatBatch attention_context(
      hidden_states.size(),
      std::vector<float>(static_cast<std::size_t>(config->hidden_size), 0.0f));
    // Dequantize this layer's KV ONCE instead of once per (query, head, source).
    //
    // The default KV mode is f16, so row_f32 converts a head_dim row on every
    // call. Decode calls it once per row per token, which is why f16 measured
    // free there. Prefill's attention is O(tokens^2 * heads): at 269 tokens that
    // is ~1.7M calls per layer, ~49M per prefill, each converting 128 halves -
    // so the f16 default multiplies prefill dequant work by the sequence length.
    // Measured before this change: attention was 22.3-24.8 s of a 29-33 s
    // prefill (68-70%), running 12.5 GFLOP at 0.56 GFLOP/s.
    //
    // One call per layer covers the whole cache, and for f32 mode row_f32
    // returns a direct pointer, so that mode copies nothing. Values are the same
    // either way - this is the identical dequantization, hoisted.
    const std::size_t kv_plain_elements =
      static_cast<std::size_t>(final_cached_positions) * static_cast<std::size_t>(kv_hidden_size);
    const float *keys_source =
      layer_cache.keys.row_f32(0, kv_plain_elements, keys_dequant_scratch);
    const float *values_source =
      layer_cache.values.row_f32(0, kv_plain_elements, values_dequant_scratch);

    // Re-lay the cache head-major: [kv_head][position][head_dim].
    //
    // The cache is stored position-major, so scanning one head's keys across
    // sources strides by kv_hidden_size - 4 KB on this model - and every 512-byte
    // row lands in a different cache line set. Attention touches ~1.8 GB per
    // layer that way, which is why it is bandwidth-bound rather than
    // compute-bound: the AVX2 build measured no faster than the scalar one here
    // (4950-5650 ms vs 4806-4946), and vectorizing a stalled load does nothing.
    // Head-major makes the source scan sequential, and the transpose is one pass
    // over ~1.1 MB per layer against the 1.8 GB it saves.
    const std::size_t head_dim_size = static_cast<std::size_t>(config->head_dim);
    const std::size_t position_count = static_cast<std::size_t>(final_cached_positions);
    keys_head_major.resize(kv_plain_elements);
    values_head_major.resize(kv_plain_elements);
    for (std::size_t position = 0; position < position_count; position += 1) {
      for (std::uint64_t kv_head = 0; kv_head < config->kv_head_count; kv_head += 1) {
        const std::size_t source_offset =
          position * static_cast<std::size_t>(kv_hidden_size)
          + static_cast<std::size_t>(kv_head) * head_dim_size;
        const std::size_t target_offset =
          (static_cast<std::size_t>(kv_head) * position_count + position) * head_dim_size;
        std::memcpy(
          keys_head_major.data() + target_offset,
          keys_source + source_offset,
          head_dim_size * sizeof(float));
        std::memcpy(
          values_head_major.data() + target_offset,
          values_source + source_offset,
          head_dim_size * sizeof(float));
      }
    }
    const float *keys_plain = keys_head_major.data();
    const float *values_plain = values_head_major.data();

    // Head-parallel, the same shape the three decode attention paths already use
    // (persistent_row_workers over config->head_count). Prefill was the one
    // attention path left serial, and it is the larger one: decode attends one
    // query per token, prefill attends N of them, so this loop is O(N^2 * heads)
    // where decode's is O(N * heads). At 269 tokens it measured 4.3-4.5 s, ~32%
    // of prefill, on a single core while the pool sat idle.
    //
    // Splitting by head is what keeps the result bit-identical: each head owns a
    // disjoint [head * head_dim, (head + 1) * head_dim) slice of every query's
    // context vector, so there is no shared write, and within a head every query
    // still walks its sources in ascending order exactly as before.
    const auto compute_attention_heads = [&](std::uint64_t head_begin, std::uint64_t head_end) {
      // Per-worker, not shared: score_scratch cannot be reused across threads.
      std::vector<float> worker_scores;
      for (std::uint64_t head = head_begin; head < head_end; head += 1) {
        const std::uint64_t kv_head = head * config->kv_head_count / config->head_count;
        const std::size_t query_offset = static_cast<std::size_t>(head * config->head_dim);
        // Head-major: this head's rows are contiguous, so the source scan below
        // walks memory sequentially instead of striding by kv_hidden_size.
        const float *head_keys =
          keys_plain + static_cast<std::size_t>(kv_head) * position_count * head_dim_size;
        const float *head_values =
          values_plain + static_cast<std::size_t>(kv_head) * position_count * head_dim_size;
        for (std::size_t batch_index = 0; batch_index < hidden_states.size(); batch_index += 1) {
          // Causal by prefix length: position i attends cache[0..i] only.
          const std::uint64_t cached_positions =
            initial_cached_positions + static_cast<std::uint64_t>(batch_index) + 1;
          const float *query_ptr =
            queries[batch_index].data() + static_cast<std::ptrdiff_t>(query_offset);
          worker_scores.assign(static_cast<std::size_t>(cached_positions), 0.0f);
          for (std::uint64_t source = 0; source < cached_positions; source += 1) {
            const float dot = attention_dot(
              query_ptr,
              head_keys + static_cast<std::size_t>(source) * head_dim_size,
              head_dim_size);
            worker_scores[static_cast<std::size_t>(source)] = dot * attention_scale;
          }
          softmax_in_place(worker_scores);
          float *context_ptr =
            attention_context[batch_index].data() + static_cast<std::ptrdiff_t>(query_offset);
          for (std::uint64_t source = 0; source < cached_positions; source += 1) {
            attention_axpy(
              context_ptr,
              head_values + static_cast<std::size_t>(source) * head_dim_size,
              worker_scores[static_cast<std::size_t>(source)],
              head_dim_size);
          }
        }
      }
    };
    const std::uint64_t attention_worker_count = choose_row_worker_count(
      config->head_count * static_cast<std::uint64_t>(hidden_states.size())
        * static_cast<std::uint64_t>(final_cached_positions) * config->head_dim,
      config->head_count,
      16384,
      1);
    if (attention_worker_count <= 1) {
      compute_attention_heads(0, config->head_count);
    } else {
      persistent_row_workers().execute(
        config->head_count, attention_worker_count, compute_attention_heads);
    }

    record_decode_phase_metric(
      session, "prefill-attention-core", static_cast<std::int64_t>(layer_number),
      attention_stage_started, prefill_stage_gpu_count, prefill_stage_gpu_detail);

    const auto output_stage_started = std::chrono::steady_clock::now();
    FloatBatch attention_outputs = matrix_batch_mul_tensor(
      state,
      tensor_at(state, layer_index.output),
      shapes.output_shape,
      attention_context,
      execution_config,
      "Llama batched attention output projection");
    record_decode_phase_metric(
      session, "prefill-attn-output", static_cast<std::int64_t>(layer_number),
      output_stage_started, prefill_stage_gpu_count, prefill_stage_gpu_detail);

    const auto ffn_stage_started = std::chrono::steady_clock::now();
    FloatBatch residuals;
    residuals.reserve(hidden_states.size());
    FloatBatch ffn_normed;
    ffn_normed.reserve(hidden_states.size());
    for (std::size_t batch_index = 0; batch_index < hidden_states.size(); batch_index += 1) {
      residuals.push_back(add_vectors(
        hidden_states[batch_index],
        attention_outputs[batch_index],
        "Llama batched attention residual"));
      ffn_normed.push_back(rms_norm(residuals.back(), ffn_norm_weights));
    }

    FloatBatch down;
    if (!try_ffn_chain_batch_resident(
      state,
      tensor_at(state, layer_index.gate),
      shapes.gate_shape,
      tensor_at(state, layer_index.up),
      shapes.up_shape,
      tensor_at(state, layer_index.down),
      shapes.down_shape,
      ffn_normed,
      execution_config,
      down,
      true)) {
      FloatBatch gates = matrix_batch_mul_tensor(
        state,
        tensor_at(state, layer_index.gate),
        shapes.gate_shape,
        ffn_normed,
        execution_config,
        "Llama batched MLP gate projection");
      const FloatBatch ups = matrix_batch_mul_tensor(
        state,
        tensor_at(state, layer_index.up),
        shapes.up_shape,
        ffn_normed,
        execution_config,
        "Llama batched MLP up projection");
      for (std::size_t batch_index = 0; batch_index < gates.size(); batch_index += 1) {
        if (gates[batch_index].size() != ups[batch_index].size()) {
          throw WorkerError("NATIVE_WORKER_ERROR", "Llama batched MLP gate and up projections produced mismatched shapes.");
        }
        for (std::size_t item = 0; item < gates[batch_index].size(); item += 1) {
          gates[batch_index][item] = silu(gates[batch_index][item]) * ups[batch_index][item];
        }
      }
      down = matrix_batch_mul_tensor(
        state,
        tensor_at(state, layer_index.down),
        shapes.down_shape,
        gates,
        execution_config,
        "Llama batched MLP down projection");
    }
    for (std::size_t batch_index = 0; batch_index < hidden_states.size(); batch_index += 1) {
      hidden_states[batch_index] = add_vectors(
        residuals[batch_index],
        down[batch_index],
        "Llama batched MLP residual");
    }
    record_decode_phase_metric(
      session, "prefill-ffn", static_cast<std::int64_t>(layer_number),
      ffn_stage_started, prefill_stage_gpu_count, prefill_stage_gpu_detail);
  }
  if (execution_config != nullptr) {
    execution_config->current_layer_index = -1;
  }

  session.sequence_token_ids.insert(
    session.sequence_token_ids.end(),
    token_ids.begin(),
    token_ids.end());
  session.cached_token_count += static_cast<std::uint64_t>(token_ids.size());
  session.last_hidden_state = rms_norm(
    hidden_states.back(),
    cached_tensor_vector_f32(state, state.tensor_index.output_norm, "Llama output norm weights"));
  session.last_logits = project_hidden_to_logits(
    state,
    tensor_at(state, state.tensor_index.output),
    session.last_hidden_state);
  // Speculative verification needs the next-token distribution after every
  // fed position, not just the last one. The vocab projection per position is
  // cheap next to the layer stack the batch already paid for.
  if (all_position_logits != nullptr) {
    all_position_logits->clear();
    all_position_logits->reserve(hidden_states.size());
    for (std::size_t index = 0; index + 1 < hidden_states.size(); index += 1) {
      const std::vector<float> normed = rms_norm(
        hidden_states[index],
        cached_tensor_vector_f32(state, state.tensor_index.output_norm, "Llama output norm weights"));
      all_position_logits->push_back(project_hidden_to_logits(
        state,
        tensor_at(state, state.tensor_index.output),
        normed));
    }
    all_position_logits->push_back(session.last_logits);
  }
}

static void execute_llama_decode_batch(
  const LoadedModelState &state,
  const std::vector<SessionState *> &sessions,
  const std::vector<std::uint64_t> &token_ids,
  BatchExecutionConfig *execution_config,
  FloatBatch *row_logits) {
  if (sessions.empty() || sessions.size() != token_ids.size()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Batched decode requires one token per session.");
  }
  const LlamaExecutionConfig *config = cached_llama_execution_config(state);
  if (config == nullptr) {
    throw WorkerError("NATIVE_EXECUTOR_UNIMPLEMENTED", "The loaded GGUF is missing the attention metadata required for native Llama execution.");
  }
  if (!state.llama_layer_shapes_ready
    || state.llama_layer_shapes.size() != state.tensor_index.layers.size()) {
    throw WorkerError("NATIVE_EXECUTOR_UNIMPLEMENTED", "Batched decode requires validated layer shape configurations.");
  }

  const std::size_t layer_count = state.tensor_index.layers.size();
  const std::size_t batch_size = sessions.size();
  const std::uint64_t kv_hidden_size = config->kv_head_count * config->head_dim;
  const float attention_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(config->head_dim)));

  // Each row carries its own absolute position, because the sequences in a
  // batch are at unrelated points in their own generations. Using the row index
  // here -- which is correct for a single-sequence batch and is what the
  // prefill path does -- would apply the wrong RoPE rotation to every row but
  // the first, and produce fluent, wrong text rather than an error.
  std::vector<std::uint64_t> row_positions(batch_size, 0);
  for (std::size_t row = 0; row < batch_size; row += 1) {
    if (sessions[row] == nullptr) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Batched decode received a null session.");
    }
    if (sessions[row]->layered_kv_cache.size() != layer_count) {
      sessions[row]->layered_kv_cache.assign(layer_count, SessionState::LayerKvCache {});
    }
    row_positions[row] = sessions[row]->cached_token_count;
  }

  FloatBatch hidden_states;
  hidden_states.reserve(batch_size);
  for (const std::uint64_t token_id : token_ids) {
    hidden_states.push_back(tensor_row_f32(
      state,
      tensor_at(state, state.tensor_index.token_embeddings),
      token_id));
  }

  for (std::size_t layer_number = 0; layer_number < layer_count; layer_number += 1) {
    if (execution_config != nullptr) {
      execution_config->current_layer_index = static_cast<std::int64_t>(layer_number);
    }
    const LlamaLayerTensorIndex &layer_index = state.tensor_index.layers[layer_number];
    const LlamaLayerShapeConfig &shapes = state.llama_layer_shapes[layer_number];

    const std::vector<float> &attention_norm_weights = cached_tensor_vector_f32(
      state, layer_index.attention_norm, "Llama attention norm weights");
    const std::vector<float> &ffn_norm_weights = cached_tensor_vector_f32(
      state, layer_index.ffn_norm, "Llama FFN norm weights");

    FloatBatch normed;
    normed.reserve(batch_size);
    for (const std::vector<float> &hidden_state : hidden_states) {
      normed.push_back(rms_norm(hidden_state, attention_norm_weights));
    }

    // THE WIN: one traversal of this layer's Q/K/V weights serves every
    // sequence in the batch. Identical call to the single-session path, which
    // is what keeps B=1 byte-identical.
    FloatBatch queries;
    FloatBatch keys;
    FloatBatch values;
    if (!try_fused_qkv_batch_resident(
      state,
      tensor_at(state, layer_index.query), shapes.query_shape,
      tensor_at(state, layer_index.key), shapes.key_shape,
      tensor_at(state, layer_index.value), shapes.value_shape,
      normed, execution_config, queries, keys, values)) {
      queries = matrix_batch_mul_tensor(
        state, tensor_at(state, layer_index.query), shapes.query_shape,
        normed, execution_config, "Llama batched decode query projection");
      keys = matrix_batch_mul_tensor(
        state, tensor_at(state, layer_index.key), shapes.key_shape,
        normed, execution_config, "Llama batched decode key projection");
      values = matrix_batch_mul_tensor(
        state, tensor_at(state, layer_index.value), shapes.value_shape,
        normed, execution_config, "Llama batched decode value projection");
    }

    for (std::size_t row = 0; row < batch_size; row += 1) {
      apply_rope_in_place(
        queries[row], row_positions[row],
        config->head_count, config->head_dim,
        config->rope_freq_base, config->rope_freq_scale, config->rope_freq_factors);
      apply_rope_in_place(
        keys[row], row_positions[row],
        config->kv_head_count, config->head_dim,
        config->rope_freq_base, config->rope_freq_scale, config->rope_freq_factors);
      append_layer_cache(
        sessions[row]->layered_kv_cache[layer_number], keys[row], values[row]);
    }

    // Attention stays per sequence. There is no shared read to amortise here --
    // every row walks its own KV -- and the lengths are ragged, so a merged
    // kernel would be slower as well as harder to prove correct.
    FloatBatch attention_context(
      batch_size, std::vector<float>(static_cast<std::size_t>(config->hidden_size), 0.0f));
    for (std::size_t row = 0; row < batch_size; row += 1) {
      const SessionState::LayerKvCache &layer_cache =
        sessions[row]->layered_kv_cache[layer_number];
      const std::uint64_t cached_positions = row_positions[row] + 1;
      if (layer_cache.keys.size() < static_cast<std::size_t>(cached_positions * kv_hidden_size)) {
        throw WorkerError("NATIVE_WORKER_ERROR", "Batched decode KV cache is shorter than the session position count.");
      }
      std::vector<float> kv_row_scratch;
      for (std::uint64_t head = 0; head < config->head_count; head += 1) {
        const std::uint64_t kv_head = head * config->kv_head_count / config->head_count;
        const std::size_t query_offset = static_cast<std::size_t>(head * config->head_dim);
        const float *query_ptr = queries[row].data() + static_cast<std::ptrdiff_t>(query_offset);
        std::vector<float> scores(static_cast<std::size_t>(cached_positions), 0.0f);
        for (std::uint64_t source = 0; source < cached_positions; source += 1) {
          const std::size_t source_base = static_cast<std::size_t>(
            source * kv_hidden_size + kv_head * config->head_dim);
          const float *key_ptr = layer_cache.keys.row_f32(
            source_base, static_cast<std::size_t>(config->head_dim), kv_row_scratch);
          scores[static_cast<std::size_t>(source)] =
            attention_dot(query_ptr, key_ptr, static_cast<std::size_t>(config->head_dim)) * attention_scale;
        }
        const std::vector<float> weights = softmax(scores);
        float *context_ptr = attention_context[row].data() + static_cast<std::ptrdiff_t>(query_offset);
        for (std::uint64_t source = 0; source < cached_positions; source += 1) {
          const std::size_t source_base = static_cast<std::size_t>(
            source * kv_hidden_size + kv_head * config->head_dim);
          const float *value_ptr = layer_cache.values.row_f32(
            source_base, static_cast<std::size_t>(config->head_dim), kv_row_scratch);
          attention_axpy(
            context_ptr, value_ptr,
            weights[static_cast<std::size_t>(source)],
            static_cast<std::size_t>(config->head_dim));
        }
      }
    }

    bool tail_fused = false;
    const bool tail_fusion_blocked = std::any_of(
      sessions.begin(),
      sessions.end(),
      [](const SessionState *session) {
        return session->decode_layer_tail_fuse_blocked;
      });
    if (!tail_fusion_blocked) {
      FloatBatch fused_hidden_states;
      if (try_llama_layer_tail_batch_resident(
          state,
          tensor_at(state, layer_index.output),
          shapes.output_shape,
          tensor_at(state, layer_index.gate),
          shapes.gate_shape,
          tensor_at(state, layer_index.up),
          shapes.up_shape,
          tensor_at(state, layer_index.down),
          shapes.down_shape,
          attention_context,
          hidden_states,
          ffn_norm_weights,
          1e-5f,
          fused_hidden_states)) {
        hidden_states = std::move(fused_hidden_states);
        tail_fused = true;
      } else {
        for (SessionState *session : sessions) {
          session->decode_layer_tail_fuse_blocked = true;
          session->decode_layer_tail_fuse_detail =
            "Direct3D 12 fused batched Llama layer tail fell back to split decode phases.";
        }
      }
    }
    if (tail_fused) {
      continue;
    }

    const FloatBatch attention_outputs = matrix_batch_mul_tensor(
      state, tensor_at(state, layer_index.output), shapes.output_shape,
      attention_context, execution_config, "Llama batched decode attention output projection");

    FloatBatch residuals;
    residuals.reserve(batch_size);
    FloatBatch ffn_normed;
    ffn_normed.reserve(batch_size);
    for (std::size_t row = 0; row < batch_size; row += 1) {
      residuals.push_back(add_vectors(
        hidden_states[row], attention_outputs[row], "Llama batched decode attention residual"));
      ffn_normed.push_back(rms_norm(residuals.back(), ffn_norm_weights));
    }

    FloatBatch down;
    if (!try_ffn_chain_batch_resident(
      state,
      tensor_at(state, layer_index.gate), shapes.gate_shape,
      tensor_at(state, layer_index.up), shapes.up_shape,
      tensor_at(state, layer_index.down), shapes.down_shape,
      ffn_normed, execution_config, down, true)) {
      FloatBatch gates = matrix_batch_mul_tensor(
        state, tensor_at(state, layer_index.gate), shapes.gate_shape,
        ffn_normed, execution_config, "Llama batched decode MLP gate projection");
      const FloatBatch ups = matrix_batch_mul_tensor(
        state, tensor_at(state, layer_index.up), shapes.up_shape,
        ffn_normed, execution_config, "Llama batched decode MLP up projection");
      for (std::size_t row = 0; row < gates.size(); row += 1) {
        if (gates[row].size() != ups[row].size()) {
          throw WorkerError("NATIVE_WORKER_ERROR", "Batched decode MLP gate and up projections produced mismatched shapes.");
        }
        for (std::size_t item = 0; item < gates[row].size(); item += 1) {
          gates[row][item] = silu(gates[row][item]) * ups[row][item];
        }
      }
      down = matrix_batch_mul_tensor(
        state, tensor_at(state, layer_index.down), shapes.down_shape,
        gates, execution_config, "Llama batched decode MLP down projection");
    }
    for (std::size_t row = 0; row < batch_size; row += 1) {
      hidden_states[row] = add_vectors(
        residuals[row], down[row], "Llama batched decode MLP residual");
    }
  }
  if (execution_config != nullptr) {
    execution_config->current_layer_index = -1;
  }

  const std::vector<float> &output_norm_weights =
    cached_tensor_vector_f32(state, state.tensor_index.output_norm, "Llama output norm weights");
  FloatBatch normalized_hidden_states;
  normalized_hidden_states.reserve(batch_size);
  for (const std::vector<float> &hidden_state : hidden_states) {
    normalized_hidden_states.push_back(rms_norm(hidden_state, output_norm_weights));
  }
  // The LM head is one of the largest tensors in the model. Projecting each
  // session independently rereads that tensor B times after the layer stack
  // has already done the work to keep every other projection shared. Keep the
  // rows together for the final vocabulary projection as well; this preserves
  // one output row per session while reading/staging the output weights once.
  const TensorInfo &output_tensor = tensor_at(state, state.tensor_index.output);
  const MatrixShape output_shape =
    tensor_matrix_shape(output_tensor, "Llama batched decode vocabulary projection");
  FloatBatch batch_logits = matrix_batch_mul_tensor(
    state,
    output_tensor,
    output_shape,
    normalized_hidden_states,
    execution_config,
    "Llama batched decode vocabulary projection");
  if (batch_logits.size() != batch_size) {
    throw WorkerError(
      "NATIVE_WORKER_ERROR",
      "Batched decode vocabulary projection returned the wrong row count.");
  }
  if (row_logits != nullptr) {
    row_logits->clear();
    row_logits->reserve(batch_size);
  }
  for (std::size_t row = 0; row < batch_size; row += 1) {
    SessionState &session = *sessions[row];
    session.sequence_token_ids.push_back(token_ids[row]);
    session.cached_token_count += 1;
    session.last_hidden_state = std::move(normalized_hidden_states[row]);
    session.last_logits = std::move(batch_logits[row]);
    if (row_logits != nullptr) {
      row_logits->push_back(session.last_logits);
    }
  }
}

static std::vector<float> execute_llama_last_hidden(
  const LoadedModelState &state,
  const std::vector<std::uint64_t> &token_ids) {
  SessionState session;
  reset_layered_session_state(session, state.tensor_index.layers.size());
  for (std::uint64_t token_id : token_ids) {
    execute_llama_consume_token(state, session, token_id);
  }
  return session.last_hidden_state;
}

static void write_line(const JsonValue &value);

// Emits one decoded token as a JSON-RPC notification (no id, so clients that
// do not understand it drop the line safely). Only active when the generate
// request asked for token events.
static void emit_generate_token_event(
  const SessionState &session,
  std::uint64_t step,
  std::uint64_t token_id,
  const std::string &token_text) {
  if (!session.stream_tokens) {
    return;
  }
  write_line(JsonValue(JsonObject {
    {"jsonrpc", JsonValue("2.0")},
    {"method", JsonValue("generate.token")},
    {"params", JsonValue(JsonObject {
      {"sessionId", JsonValue(session.session_id)},
      {"index", JsonValue(step)},
      {"tokenId", JsonValue(token_id)},
      {"token", JsonValue(token_text)},
    })},
  }));
}

static JsonObject execute_zero_layer_generate(
  const LoadedModelState &state,
  SessionState &session,
  std::uint64_t max_tokens,
  std::uint64_t prefill_ms) {
  if (session.sequence_token_ids.empty()) {
    throw WorkerError(
      "NATIVE_EXECUTOR_UNIMPLEMENTED",
      "The native worker cannot execute this GGUF prompt yet because prompt token IDs are unavailable.");
  }
  const TensorInfo &token_embeddings = tensor_at(state, state.tensor_index.token_embeddings);
  const TensorInfo &output = tensor_at(state, state.tensor_index.output);
  const std::vector<float> &output_norm_weights = cached_tensor_vector_f32(
    state, state.tensor_index.output_norm, "Llama output norm weights");

  const auto decode_started_at = std::chrono::steady_clock::now();
  d3d12_reset_runtime_telemetry();
  std::vector<std::uint64_t> generated_token_ids;
  JsonArray generated_token_ids_json;
  JsonArray logits_preview_json;
  std::vector<std::string> generated_tokens;
  JsonArray chunks_json;
  std::uint64_t current_token_id = session.sequence_token_ids.back();
  std::vector<float> last_hidden_preview;
  std::vector<float> last_logits;

  for (std::uint64_t step = 0; step < max_tokens; step += 1) {
    std::vector<float> hidden_state = tensor_row_f32(state, token_embeddings, current_token_id);
    hidden_state = rms_norm(hidden_state, output_norm_weights);
    last_hidden_preview = hidden_state;
    last_logits = project_hidden_to_logits(state, output, hidden_state);
    current_token_id = select_next_token(last_logits, session.sampling);
    generated_token_ids.push_back(current_token_id);
    generated_token_ids_json.push_back(JsonValue(current_token_id));
    session.sequence_token_ids.push_back(current_token_id);
    session.cached_token_count += 1;

    const std::string token_text = current_token_id < state.model.token_vocabulary.size()
      ? state.model.token_vocabulary[static_cast<std::size_t>(current_token_id)]
      : ("<token:" + std::to_string(current_token_id) + ">");
    generated_tokens.push_back(token_text);
    emit_generate_token_event(session, step, current_token_id, token_text);
  }

  for (std::size_t index = 0; index < generated_tokens.size(); index += 1) {
    chunks_json.push_back(JsonValue(generated_tokens[index]));
  }
  const std::size_t logits_preview_count = std::min<std::size_t>(8, last_logits.size());
  for (std::size_t index = 0; index < logits_preview_count; index += 1) {
    logits_preview_json.push_back(JsonValue(static_cast<double>(last_logits[index])));
  }
  JsonArray hidden_preview_json;
  const std::size_t hidden_preview_count = std::min<std::size_t>(8, last_hidden_preview.size());
  for (std::size_t index = 0; index < hidden_preview_count; index += 1) {
    hidden_preview_json.push_back(JsonValue(static_cast<double>(last_hidden_preview[index])));
  }

  const auto decode_finished_at = std::chrono::steady_clock::now();
  const std::uint64_t decode_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(decode_finished_at - decode_started_at).count()));
  const double tokens_per_second = decode_ms == 0
    ? 0.0
    : (static_cast<double>(generated_token_ids.size()) / static_cast<double>(decode_ms)) * 1000.0;
  session.generated_token_count += static_cast<std::uint64_t>(generated_token_ids.size());
  session.last_decode_token_count = static_cast<std::uint64_t>(generated_token_ids.size());
  session.last_decode_ms = decode_ms;

  std::ostringstream output_text;
  for (const std::string &token : generated_tokens) {
    output_text << token;
  }
  return {
    {"runtimeProvider", JsonValue("lizard-native")},
    {"nativeBackend", JsonValue(ggml_cpu_backend_effective_name())},
    {"nativeModelStatus", JsonValue("minimal-graph-executed")},
    {"loadMs", JsonValue(state.load_ms)},
    {"prefillMs", JsonValue(prefill_ms)},
    {"decodeMs", JsonValue(decode_ms)},
    {"tokensPerSecond", JsonValue(tokens_per_second)},
    {"kvCacheBytes", JsonValue(session.kv_cache_bytes)},
    {"output", JsonValue(output_text.str())},
    {"chunks", JsonValue(chunks_json)},
    {"generatedTokenIds", JsonValue(generated_token_ids_json)},
    {"cachedTokenCount", JsonValue(static_cast<std::uint64_t>(session.sequence_token_ids.size()))},
    {"sampling", JsonValue(JsonObject {
      {"temperature", JsonValue(session.sampling.temperature)},
      {"topK", JsonValue(session.sampling.top_k)},
      {"topP", JsonValue(session.sampling.top_p)},
      {"seed", JsonValue(session.sampling.seed)},
    })},
    {"sessionMetrics", JsonValue(build_session_metrics_object(session))},
    {"hiddenStatePreview", JsonValue(hidden_preview_json)},
    {"logitsPreview", JsonValue(logits_preview_json)},
    {"compatibility", JsonValue(state.compatibility)},
  };
}

static JsonObject execute_llama_layered_generate(
  const LoadedModelState &state,
  SessionState &session,
  std::uint64_t max_tokens,
  std::uint64_t prefill_ms) {
  if (session.sequence_token_ids.empty() || session.last_logits.empty() || !session.prefill_complete) {
    throw WorkerError(
      "NATIVE_EXECUTOR_UNIMPLEMENTED",
      "The native worker cannot execute this GGUF prompt yet because prompt token IDs are unavailable.");
  }

  const auto decode_started_at = std::chrono::steady_clock::now();
  d3d12_reset_runtime_telemetry();
  reset_decode_gpu_trace();
  session.decode_phase_metrics.clear();
  std::vector<std::uint64_t> generated_token_ids;
  JsonArray generated_token_ids_json;
  JsonArray logits_preview_json;
  JsonArray hidden_preview_json;
  std::vector<std::string> generated_tokens;
  JsonArray chunks_json;
  std::vector<float> last_hidden;
  std::vector<float> last_logits;

  // Speculative decoding is exact only for greedy sampling: every emitted
  // token is the argmax of true logits, so the sequence matches the
  // sequential loop byte-for-byte.
  const bool speculative = speculative_decode_enabled()
    && session.sampling.temperature <= 0.0
    && state.llama_layer_shapes_ready;
  std::uint64_t speculative_accepted_tokens = 0;
  std::uint64_t speculative_rounds = 0;

  // End of turn. This loop previously ran to max_tokens unconditionally - it
  // had no EOS check at all - so an instruct model's turn-ender was ignored and
  // a base model (which never emits EOS) always decoded the whole budget, long
  // after its answer was finished. Stop on the GGUF EOS and on the request's
  // stop sequences (chat turn-enders, instruction/role headers), matched
  // against the decoded text tail so a marker split across tokens is caught.
  const auto eos_token_id = json_to_u64(find_object_value(state.model.special_tokens, "eos"));
  std::string stop_scan_tail;
  const std::size_t kStopScanTailMax = 64;
  bool generation_stopped = false;

  // Emits one token and returns true when generation should stop. EOS is a
  // model control token and is never emitted, regardless of whether the caller
  // supplied extra textual stop sequences. HTTP benchmarks exercise the same
  // user-visible completion semantics as chat; decoding hidden tokens after
  // EOS would inflate runtime work and token accounting while producing no
  // additional answer. A textual stop-sequence's own tokens are emitted (the
  // streaming filter in native-cpu-runtime.js trims the marker) and only then
  // does the loop stop.
  const bool apply_text_stop = !session.stop_sequences.empty();
  const auto emit_token = [&](std::uint64_t step, std::uint64_t token_id) -> bool {
    if (eos_token_id.has_value() && token_id == eos_token_id.value()) {
      return true;
    }
    generated_token_ids.push_back(token_id);
    generated_token_ids_json.push_back(JsonValue(token_id));
    const std::string token_text = token_id < state.model.token_vocabulary.size()
      ? state.model.token_vocabulary[static_cast<std::size_t>(token_id)]
      : ("<token:" + std::to_string(token_id) + ">");
    generated_tokens.push_back(token_text);
    emit_generate_token_event(session, step, token_id, token_text);
    if (apply_text_stop) {
      stop_scan_tail += decode_bpe_whitespace(token_text);
      if (stop_scan_tail.size() > kStopScanTailMax) {
        stop_scan_tail.erase(0, stop_scan_tail.size() - kStopScanTailMax);
      }
      for (const std::string &stop_marker : session.stop_sequences) {
        if (!stop_marker.empty() && stop_scan_tail.find(stop_marker) != std::string::npos) {
          return true;
        }
      }
    }
    return false;
  };

  for (std::uint64_t step = 0; step < max_tokens && !generation_stopped;) {
    const std::uint64_t token_id =
      session.last_logits.empty() && session.last_gpu_argmax_token >= 0
      ? static_cast<std::uint64_t>(session.last_gpu_argmax_token)
      : select_next_token(session.last_logits, session.sampling);
    const bool stop_after_token = emit_token(step, token_id);
    step += 1;
    if (stop_after_token) {
      break;
    }

    std::vector<std::uint64_t> drafts;
    if (speculative && step < max_tokens && !session.last_logits.empty()) {
      std::vector<std::uint64_t> context = session.sequence_token_ids;
      context.push_back(token_id);
      drafts = draft_tokens_from_ngram(
        context,
        std::min<std::size_t>(speculative_draft_limit(), static_cast<std::size_t>(max_tokens - step)));
    }

    if (drafts.empty()) {
      execute_llama_consume_token(state, session, token_id);
    } else {
      // One batched forward feeds the sampled token plus the drafts and
      // returns the true next-token logits after every fed position.
      std::vector<std::uint64_t> batch;
      batch.reserve(1 + drafts.size());
      batch.push_back(token_id);
      batch.insert(batch.end(), drafts.begin(), drafts.end());
      const std::uint64_t cached_before = session.cached_token_count;
      FloatBatch position_logits;
      // EnhancementRun_04 Phase 9. The verify pass MUST carry the GPU lane.
      // With a null config it ran entirely on the CPU while ordinary decode ran
      // on the iGPU, so every speculative round paid a CPU forward pass to
      // check tokens the GPU had produced -- measured 0.15-0.34x, a 3-7x
      // slowdown that made a lossless technique look like a broken one.
      // `min_gpu_prefill_tokens` is 1 because that threshold is a prefill
      // heuristic and a draft batch is a handful of rows by nature.
      BatchExecutionConfig verify_execution;
      verify_execution.allow_hybrid_gpu = gpu_decode_lane_enabled_for_model(state)
        && d3d12_batch_matmul_capabilities().supported;
      verify_execution.min_gpu_prefill_tokens = 1;
      execute_llama_prefill_batch(state, session, batch, &verify_execution, &position_logits);
      speculative_rounds += 1;

      // Accept drafts while they match the true greedy continuation.
      std::size_t accepted = 0;
      while (accepted < drafts.size() && step < max_tokens) {
        const std::uint64_t true_next = static_cast<std::uint64_t>(
          argmax_index(position_logits[accepted]));
        if (true_next != drafts[accepted]) {
          break;
        }
        const bool stop_after_draft = emit_token(step, true_next);
        step += 1;
        accepted += 1;
        speculative_accepted_tokens += 1;
        if (stop_after_draft) {
          generation_stopped = true;
          break;
        }
      }

      // Drop the KV entries of rejected drafts; logits after the last kept
      // position drive the next iteration's sampling, so a mismatch costs
      // nothing (the corrected token is that argmax).
      truncate_session_kv_positions(session, cached_before + 1 + accepted);
      session.last_logits = position_logits[accepted];
    }

    last_hidden = session.last_hidden_state;
    last_logits = session.last_logits;
  }
  if (speculative_rounds > 0) {
    session.decode_backend_detail = "speculative n-gram: "
      + std::to_string(speculative_accepted_tokens) + " of "
      + std::to_string(generated_token_ids.size()) + " tokens accepted across "
      + std::to_string(speculative_rounds) + " rounds";
  }

  for (const std::string &token : generated_tokens) {
    chunks_json.push_back(JsonValue(token));
  }
  const std::size_t logits_preview_count = std::min<std::size_t>(8, last_logits.size());
  for (std::size_t index = 0; index < logits_preview_count; index += 1) {
    logits_preview_json.push_back(JsonValue(static_cast<double>(last_logits[index])));
  }
  const std::size_t hidden_preview_count = std::min<std::size_t>(8, last_hidden.size());
  for (std::size_t index = 0; index < hidden_preview_count; index += 1) {
    hidden_preview_json.push_back(JsonValue(static_cast<double>(last_hidden[index])));
  }

  const auto decode_finished_at = std::chrono::steady_clock::now();
  const std::uint64_t decode_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(decode_finished_at - decode_started_at).count()));
  const double tokens_per_second = decode_ms == 0
    ? 0.0
    : (static_cast<double>(generated_token_ids.size()) / static_cast<double>(decode_ms)) * 1000.0;
  session.generated_token_count += static_cast<std::uint64_t>(generated_token_ids.size());
  session.last_decode_token_count = static_cast<std::uint64_t>(generated_token_ids.size());
  session.last_decode_ms = decode_ms;
  session.decode_backend = g_decode_gpu_used
    ? d3d12_batch_matmul_capabilities().backend_name
    : ggml_cpu_backend_effective_name();
  session.decode_backend_detail = g_decode_gpu_detail;
  if (!g_decode_gpu_used
    && gpu_decode_lane_requested()
    && !gpu_decode_model_quantization_eligible(state)) {
    session.decode_backend_detail = gpu_decode_quantization_gate_detail(state);
  }
  if (!g_decode_gpu_used && session.decode_backend_detail.empty()) {
    session.decode_backend_detail = gpu_decode_disabled_detail();
  }
  session.native_backend = session.decode_backend;
  session.native_backend_detail = session.decode_backend_detail;

  std::ostringstream output_text;
  for (const std::string &token : generated_tokens) {
    output_text << token;
  }
  return {
    {"runtimeProvider", JsonValue("lizard-native")},
    {"nativeBackend", JsonValue(ggml_cpu_backend_effective_name())},
    {"nativeModelStatus", JsonValue("llama-graph-executed")},
    {"loadMs", JsonValue(state.load_ms)},
    {"prefillMs", JsonValue(prefill_ms)},
    {"decodeMs", JsonValue(decode_ms)},
    {"tokensPerSecond", JsonValue(tokens_per_second)},
    {"kvCacheBytes", JsonValue(session.kv_cache_bytes)},
    {"output", JsonValue(output_text.str())},
    {"chunks", JsonValue(chunks_json)},
    {"generatedTokenIds", JsonValue(generated_token_ids_json)},
    {"cachedTokenCount", JsonValue(session.cached_token_count)},
    {"sampling", JsonValue(JsonObject {
      {"temperature", JsonValue(session.sampling.temperature)},
      {"topK", JsonValue(session.sampling.top_k)},
      {"topP", JsonValue(session.sampling.top_p)},
      {"seed", JsonValue(session.sampling.seed)},
    })},
    {"sessionMetrics", JsonValue(build_session_metrics_object(session))},
    {"hiddenStatePreview", JsonValue(hidden_preview_json)},
    {"logitsPreview", JsonValue(logits_preview_json)},
    {"compatibility", JsonValue(state.compatibility)},
  };
}

// Decode loop for the family expansion path. Mirrors the llama loop but
// consumes tokens through the parameterized family graph.
static JsonObject execute_family_layered_generate(
  const LoadedModelState &state,
  SessionState &session,
  std::uint64_t max_tokens,
  std::uint64_t prefill_ms) {
  if (session.sequence_token_ids.empty() || session.last_logits.empty() || !session.prefill_complete) {
    throw WorkerError(
      "NATIVE_EXECUTOR_UNIMPLEMENTED",
      "The native worker cannot execute this GGUF prompt yet because prompt token IDs are unavailable.");
  }
  const auto spec = resolve_family_spec(state.model);
  const std::string family = spec.has_value() ? spec->family : "family";

  const auto decode_started_at = std::chrono::steady_clock::now();
  session.decode_phase_metrics.clear();
  std::vector<std::uint64_t> generated_token_ids;
  JsonArray generated_token_ids_json;
  JsonArray logits_preview_json;
  JsonArray hidden_preview_json;
  std::vector<std::string> generated_tokens;
  JsonArray chunks_json;
  std::vector<float> last_hidden;
  std::vector<float> last_logits;

  // End of turn, same rules as the llama and Gemma loops: the GGUF base EOS
  // is consumed silently, a textual stop marker (<|im_end|>, <|eot_id|>, ...)
  // is emitted so the streaming filter can trim it, and either one ends the
  // loop. Without this every answer decoded the whole max_tokens budget.
  const auto eos_token_id = json_to_u64(find_object_value(state.model.special_tokens, "eos"));
  std::string stop_scan_tail;
  const std::size_t kStopScanTailMax = 64;

  for (std::uint64_t step = 0; step < max_tokens; step += 1) {
    const std::uint64_t token_id = select_next_token(session.last_logits, session.sampling);
    const std::string token_text = token_id < state.model.token_vocabulary.size()
      ? state.model.token_vocabulary[static_cast<std::size_t>(token_id)]
      : ("<token:" + std::to_string(token_id) + ">");

    if ((eos_token_id.has_value() && token_id == eos_token_id.value()) || token_text == "<turn|>") {
      break;
    }

    generated_token_ids.push_back(token_id);
    generated_token_ids_json.push_back(JsonValue(token_id));
    generated_tokens.push_back(token_text);
    emit_generate_token_event(session, step, token_id, token_text);

    if (!session.stop_sequences.empty()) {
      stop_scan_tail += decode_bpe_whitespace(token_text);
      if (stop_scan_tail.size() > kStopScanTailMax) {
        stop_scan_tail.erase(0, stop_scan_tail.size() - kStopScanTailMax);
      }
      bool stop_sequence_hit = false;
      for (const std::string &stop_marker : session.stop_sequences) {
        if (!stop_marker.empty() && stop_scan_tail.find(stop_marker) != std::string::npos) {
          stop_sequence_hit = true;
          break;
        }
      }
      if (stop_sequence_hit) {
        break;
      }
    }

    execute_family_consume_token(state, session, token_id);
    last_hidden = session.last_hidden_state;
    last_logits = session.last_logits;
  }

  for (const std::string &token : generated_tokens) {
    chunks_json.push_back(JsonValue(token));
  }
  const std::size_t logits_preview_count = std::min<std::size_t>(8, last_logits.size());
  for (std::size_t index = 0; index < logits_preview_count; index += 1) {
    logits_preview_json.push_back(JsonValue(static_cast<double>(last_logits[index])));
  }
  const std::size_t hidden_preview_count = std::min<std::size_t>(8, last_hidden.size());
  for (std::size_t index = 0; index < hidden_preview_count; index += 1) {
    hidden_preview_json.push_back(JsonValue(static_cast<double>(last_hidden[index])));
  }

  const auto decode_finished_at = std::chrono::steady_clock::now();
  const std::uint64_t decode_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(decode_finished_at - decode_started_at).count()));
  const double tokens_per_second = decode_ms == 0
    ? 0.0
    : (static_cast<double>(generated_token_ids.size()) / static_cast<double>(decode_ms)) * 1000.0;
  session.generated_token_count += static_cast<std::uint64_t>(generated_token_ids.size());
  session.last_decode_token_count = static_cast<std::uint64_t>(generated_token_ids.size());
  session.last_decode_ms = decode_ms;

  std::ostringstream output_text;
  for (const std::string &token : generated_tokens) {
    output_text << token;
  }
  return {
    {"runtimeProvider", JsonValue("lizard-native")},
    {"nativeBackend", JsonValue(ggml_cpu_backend_effective_name())},
    {"nativeModelStatus", JsonValue(family + "-graph-executed")},
    {"family", JsonValue(family)},
    {"loadMs", JsonValue(state.load_ms)},
    {"prefillMs", JsonValue(prefill_ms)},
    {"decodeMs", JsonValue(decode_ms)},
    {"tokensPerSecond", JsonValue(tokens_per_second)},
    {"kvCacheBytes", JsonValue(session.kv_cache_bytes)},
    {"output", JsonValue(output_text.str())},
    {"chunks", JsonValue(chunks_json)},
    {"generatedTokenIds", JsonValue(generated_token_ids_json)},
    {"cachedTokenCount", JsonValue(session.cached_token_count)},
    {"sampling", JsonValue(JsonObject {
      {"temperature", JsonValue(session.sampling.temperature)},
      {"topK", JsonValue(session.sampling.top_k)},
      {"topP", JsonValue(session.sampling.top_p)},
      {"seed", JsonValue(session.sampling.seed)},
    })},
    {"sessionMetrics", JsonValue(build_session_metrics_object(session))},
    {"hiddenStatePreview", JsonValue(hidden_preview_json)},
    {"logitsPreview", JsonValue(logits_preview_json)},
    {"compatibility", JsonValue(state.compatibility)},
  };
}

static JsonObject execute_gemma_layered_generate(
  const LoadedModelState &state,
  SessionState &session,
  std::uint64_t max_tokens,
  std::uint64_t prefill_ms,
  bool cache_final_token) {
  if (session.sequence_token_ids.empty() || session.last_logits.empty() || !session.prefill_complete) {
    throw WorkerError(
      "NATIVE_EXECUTOR_UNIMPLEMENTED",
      "The native Gemma worker cannot execute this GGUF prompt yet because prompt token IDs are unavailable.");
  }

  const auto decode_started_at = std::chrono::steady_clock::now();
  d3d12_reset_runtime_telemetry();
  reset_decode_gpu_trace();
  session.decode_phase_metrics.clear();
  std::vector<std::uint64_t> generated_token_ids;
  JsonArray generated_token_ids_json;
  JsonArray logits_preview_json;
  JsonArray hidden_preview_json;
  std::vector<std::string> generated_tokens;
  JsonArray chunks_json;
  std::vector<float> last_hidden;
  std::vector<float> last_logits;
  bool final_token_cached = true;
  // Base EOS from GGUF metadata (tokenizer.ggml.eos_token_id), looked up
  // once per generate call rather than per step.
  const auto eos_token_id = json_to_u64(find_object_value(state.model.special_tokens, "eos"));

  // Speculative decoding is exact only for greedy sampling (every emitted
  // token is the argmax of true logits). The batched verify runs on the CPU
  // layer stack, so it must not run while any layer's KV lives on the GPU:
  // execute_gemma_prefill_batch appends CPU KV only, and a later token graph
  // would read a GPU cache that never saw those positions. The per-round
  // gpu-KV check below keeps the lane exact on both sides of that boundary.
  const bool speculative = speculative_decode_enabled()
    && session.sampling.temperature <= 0.0
    && state.gemma_layer_configs_ready
    && state.gemma_layer_configs.size() == state.tensor_index.layers.size();
  std::uint64_t speculative_proposed_tokens = 0;
  std::uint64_t speculative_accepted_tokens = 0;
  std::uint64_t speculative_rounds = 0;
  bool speculative_stop_hit = false;

  // Rolling tail of decoded text, for matching multi-token stop sequences
  // (session.stop_sequences). Only the last few dozen characters can contain a
  // marker, so the buffer stays tiny.
  std::string stop_scan_tail;
  const std::size_t kStopScanTailMax = 64;

  for (std::uint64_t step = 0; step < max_tokens && !speculative_stop_hit; step += 1) {
    const std::uint64_t token_id =
      session.last_logits.empty() && session.last_gpu_argmax_token >= 0
      ? static_cast<std::uint64_t>(session.last_gpu_argmax_token)
      : select_next_token(session.last_logits, session.sampling);
    const std::string token_text = token_id < state.model.token_vocabulary.size()
      ? state.model.token_vocabulary[static_cast<std::size_t>(token_id)]
      : ("<token:" + std::to_string(token_id) + ">");

    // Stop as soon as the model signals its turn is done: the base EOS
    // token, or this project's own Gemma4 chat-template closing marker
    // "<turn|>" (renderGemmaChatPrompt in src/runtime/tokenizer/index.js
    // prompts with "<|turn>model\n", and the model mirrors that pattern
    // to close its own turn). Before this check, the loop ALWAYS ran to
    // max_tokens no matter what was generated - a short max_tokens just
    // cut the answer off, but a longer one let the model wander into a
    // fully hallucinated, unrelated continuation after its real answer
    // was already complete (found 2026-07-07 via the onboarding chat
    // dialog once maxTokens was raised from 200 to 512). Neither this
    // token nor anything after it is recorded - it is consumed, not
    // shown, matching standard stop-token handling.
    if ((eos_token_id.has_value() && token_id == eos_token_id.value()) || token_text == "<turn|>") {
      break;
    }

    // Extra stop sequences from the request (chat turn-enders and instruction
    // headers). The GGUF EOS above stops instruct models whose metadata marks
    // the turn-ender as EOS; this covers the ones where it does not, and base
    // models that never emit EOS. The marker's own tokens are not suppressed
    // here - the streaming layer (createStopSequenceFilter in
    // native-cpu-runtime.js) holds them back from display - so this only has to
    // stop the loop from decoding the rest of the budget after the turn ends.
    if (!session.stop_sequences.empty()) {
      // token_text is raw byte-level BPE; decode the whitespace markers so text
      // stop strings ("### ", "\nUser:") match. Special-token markers
      // (<|eot_id|>, <|im_end|>) are literal in the vocab and pass through
      // unchanged, so instruct models stop on their own turn-ender too.
      stop_scan_tail += decode_bpe_whitespace(token_text);
      if (stop_scan_tail.size() > kStopScanTailMax) {
        stop_scan_tail.erase(0, stop_scan_tail.size() - kStopScanTailMax);
      }
      bool stop_sequence_hit = false;
      for (const std::string &stop_marker : session.stop_sequences) {
        if (!stop_marker.empty() && stop_scan_tail.find(stop_marker) != std::string::npos) {
          stop_sequence_hit = true;
          break;
        }
      }
      if (stop_sequence_hit) {
        // Record this token (it is part of the visible answer up to the marker;
        // the JS filter trims the marker itself) and stop.
        generated_token_ids.push_back(token_id);
        generated_token_ids_json.push_back(JsonValue(token_id));
        generated_tokens.push_back(token_text);
        emit_generate_token_event(session, step, token_id, token_text);
        break;
      }
    }

    generated_token_ids.push_back(token_id);
    generated_token_ids_json.push_back(JsonValue(token_id));
    generated_tokens.push_back(token_text);
    emit_generate_token_event(session, step, token_id, token_text);

    session.sequence_token_ids.push_back(token_id);
    const bool need_next_logits = cache_final_token || step + 1 < max_tokens;
    if (!need_next_logits) {
      final_token_cached = false;
      last_hidden = session.last_hidden_state;
      last_logits = session.last_logits;
      continue;
    }

    // Prompt-lookup drafts (the token just emitted is already in
    // sequence_token_ids, so the n-gram context is the sequence itself).
    // GPU KV can activate on the first consume of a generation, so the
    // boundary is re-checked every round, not once up front.
    std::vector<std::uint64_t> drafts;
    if (speculative && step + 1 < max_tokens && !session.last_logits.empty()
      && std::none_of(
        session.gpu_kv_cache_active.begin(),
        session.gpu_kv_cache_active.end(),
        [](bool active) { return active; })) {
      drafts = draft_tokens_from_ngram(
        session.sequence_token_ids,
        std::min<std::size_t>(
          speculative_draft_limit(),
          static_cast<std::size_t>(max_tokens - step - 1)));
    }

    if (drafts.empty()) {
      session.sequence_token_ids.pop_back();
      execute_gemma_consume_token(state, session, token_id);
      last_hidden = session.last_hidden_state;
      last_logits = session.last_logits;
      continue;
    }

    // One batched forward feeds the sampled token plus the drafts and
    // returns the true next-token logits after every fed position.
    session.sequence_token_ids.pop_back();
    std::vector<std::uint64_t> batch;
    batch.reserve(1 + drafts.size());
    batch.push_back(token_id);
    batch.insert(batch.end(), drafts.begin(), drafts.end());
    const std::uint64_t cached_before = session.cached_token_count;
    FloatBatch position_logits;
    execute_gemma_prefill_batch(state, session, batch, nullptr, &position_logits);
    speculative_rounds += 1;
    speculative_proposed_tokens += static_cast<std::uint64_t>(drafts.size());

    // Accept drafts while they match the true greedy continuation. Unlike
    // the llama lane, every accepted token still owes the stop check: an
    // accepted EOS/"<turn|>" ends the answer and is consumed, not shown.
    std::size_t accepted = 0;
    while (accepted < drafts.size() && step + 1 < max_tokens) {
      const std::uint64_t true_next = static_cast<std::uint64_t>(
        argmax_index(position_logits[accepted]));
      if (true_next != drafts[accepted]) {
        break;
      }
      const std::string draft_text = true_next < state.model.token_vocabulary.size()
        ? state.model.token_vocabulary[static_cast<std::size_t>(true_next)]
        : ("<token:" + std::to_string(true_next) + ">");
      if ((eos_token_id.has_value() && true_next == eos_token_id.value()) || draft_text == "<turn|>") {
        speculative_stop_hit = true;
        break;
      }
      step += 1;
      generated_token_ids.push_back(true_next);
      generated_token_ids_json.push_back(JsonValue(true_next));
      generated_tokens.push_back(draft_text);
      emit_generate_token_event(session, step, true_next, draft_text);
      accepted += 1;
      speculative_accepted_tokens += 1;
    }

    // Drop the KV entries of rejected drafts; the logits after the last
    // kept position drive the next iteration's sampling, so a mismatch
    // costs nothing (the corrected token is that argmax). The batch pass
    // already appended every fed token to sequence_token_ids, so the
    // truncate (a no-op when all drafts were accepted) rewinds both the
    // KV cache and the sequence to exactly the accepted prefix.
    truncate_session_kv_positions(session, cached_before + 1 + accepted);
    session.last_logits = position_logits[accepted];
    last_hidden = session.last_hidden_state;
    last_logits = session.last_logits;
  }

  for (const std::string &token : generated_tokens) {
    chunks_json.push_back(JsonValue(token));
  }
  const std::size_t logits_preview_count = std::min<std::size_t>(8, last_logits.size());
  for (std::size_t index = 0; index < logits_preview_count; index += 1) {
    logits_preview_json.push_back(JsonValue(static_cast<double>(last_logits[index])));
  }
  const std::size_t hidden_preview_count = std::min<std::size_t>(8, last_hidden.size());
  for (std::size_t index = 0; index < hidden_preview_count; index += 1) {
    hidden_preview_json.push_back(JsonValue(static_cast<double>(last_hidden[index])));
  }

  const auto decode_finished_at = std::chrono::steady_clock::now();
  const std::uint64_t decode_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(decode_finished_at - decode_started_at).count()));
  const double tokens_per_second = decode_ms == 0
    ? 0.0
    : (static_cast<double>(generated_token_ids.size()) / static_cast<double>(decode_ms)) * 1000.0;
  session.generated_token_count += static_cast<std::uint64_t>(generated_token_ids.size());
  session.last_decode_token_count = static_cast<std::uint64_t>(generated_token_ids.size());
  session.last_decode_ms = decode_ms;
  session.decode_backend = g_decode_gpu_used
    ? d3d12_batch_matmul_capabilities().backend_name
    : ggml_cpu_backend_effective_name();
  session.decode_backend_detail = g_decode_gpu_detail;
  if (!g_decode_gpu_used
    && gpu_decode_lane_requested()
    && !gpu_decode_model_quantization_eligible(state)) {
    session.decode_backend_detail = gpu_decode_quantization_gate_detail(state);
  }
  if (!g_decode_gpu_used && session.decode_backend_detail.empty()) {
    session.decode_backend_detail = gpu_decode_disabled_detail();
  }
  if (speculative_rounds > 0) {
    const std::string speculative_note = "speculative n-gram: "
      + std::to_string(speculative_accepted_tokens) + " of "
      + std::to_string(speculative_proposed_tokens) + " drafts accepted across "
      + std::to_string(speculative_rounds) + " rounds";
    session.decode_backend_detail = session.decode_backend_detail.empty()
      ? speculative_note
      : session.decode_backend_detail + "; " + speculative_note;
  }
  session.native_backend = session.decode_backend;
  session.native_backend_detail = session.decode_backend_detail;

  std::ostringstream output_text;
  for (const std::string &token : generated_tokens) {
    output_text << token;
  }
  return {
    {"runtimeProvider", JsonValue("lizard-native")},
    {"nativeBackend", JsonValue(ggml_cpu_backend_effective_name())},
    {"nativeModelStatus", JsonValue("gemma-graph-executed")},
    {"speculativeRounds", JsonValue(speculative_rounds)},
    {"speculativeProposedTokens", JsonValue(speculative_proposed_tokens)},
    {"speculativeAcceptedTokens", JsonValue(speculative_accepted_tokens)},
    {"loadMs", JsonValue(state.load_ms)},
    {"prefillMs", JsonValue(prefill_ms)},
    {"decodeMs", JsonValue(decode_ms)},
    {"tokensPerSecond", JsonValue(tokens_per_second)},
    {"kvCacheBytes", JsonValue(session.kv_cache_bytes)},
    {"output", JsonValue(output_text.str())},
    {"chunks", JsonValue(chunks_json)},
    {"generatedTokenIds", JsonValue(generated_token_ids_json)},
    {"cachedTokenCount", JsonValue(session.cached_token_count)},
    {"sequenceTokenCount", JsonValue(static_cast<std::uint64_t>(session.sequence_token_ids.size()))},
    {"finalTokenCached", JsonValue(final_token_cached)},
    {"sampling", JsonValue(JsonObject {
      {"temperature", JsonValue(session.sampling.temperature)},
      {"topK", JsonValue(session.sampling.top_k)},
      {"topP", JsonValue(session.sampling.top_p)},
      {"seed", JsonValue(session.sampling.seed)},
    })},
    {"sessionMetrics", JsonValue(build_session_metrics_object(session))},
    {"hiddenStatePreview", JsonValue(hidden_preview_json)},
    {"logitsPreview", JsonValue(logits_preview_json)},
    {"compatibility", JsonValue(state.compatibility)},
  };
}

static std::uint64_t estimate_session_native_bytes(const SessionState &session) {
  std::uint64_t total = 0;
  total += static_cast<std::uint64_t>(session.sequence_token_ids.size() * sizeof(std::uint64_t));
  total += static_cast<std::uint64_t>(session.prompt_token_ids.size() * sizeof(std::uint64_t));
  total += static_cast<std::uint64_t>(session.last_hidden_state.size() * sizeof(float));
  total += static_cast<std::uint64_t>(session.last_logits.size() * sizeof(float));
  for (const SessionState::LayerKvCache &layer : session.layered_kv_cache) {
    total += layer.keys.storage_bytes();
    total += layer.values.storage_bytes();
  }
  return total;
}

// EnhancementRun_04 Phase 0 — decode bandwidth model.
//
// Single-stream decode on this class of machine is memory-bound: every emitted
// token reads the whole weight set once. Knowing that byte count turns a tok/s
// number into a bandwidth number, which is the only form in which it can be
// compared against a roofline and against llama.cpp on equal terms.
//
// The count is analytic rather than instrumented on purpose. Summing the tensor
// directory is exact for a fully-resident decode and costs nothing at runtime,
// where a counter in the matmul entry points would perturb the very path being
// measured. `source` states which it is so a reader is never guessing.
struct DecodeBandwidthModel {
  bool ready = false;
  std::uint64_t total_tensor_bytes = 0;
  std::uint64_t embedding_tensor_bytes = 0;
  std::uint64_t weight_bytes_per_token = 0;
  std::uint64_t kv_bytes_per_token = 0;
  bool output_tied = false;
  std::string source;
};

static DecodeBandwidthModel &decode_bandwidth_model_storage() {
  static DecodeBandwidthModel model;
  return model;
}

static const DecodeBandwidthModel &decode_bandwidth_model() {
  return decode_bandwidth_model_storage();
}

static void reset_decode_bandwidth_model() {
  decode_bandwidth_model_storage() = DecodeBandwidthModel {};
}

// Bytes one KV element occupies in the lane storage for the active mode. The
// quantized lanes carry per-block scales; those are counted where the lane
// itself counts them, so this stays the payload width only and is labelled as
// an estimate rather than a measurement.
static double kv_bytes_per_element_estimate() {
  switch (kv_cache_mode()) {
    case KvCacheMode::F16:
      return 2.0;
    case KvCacheMode::Q8_0:
      // 32 int8 payload bytes + one f16 scale per 32-element block.
      return 1.0 + (2.0 / 32.0);
    case KvCacheMode::Q4_0:
      return 0.5 + (2.0 / 32.0);
    case KvCacheMode::Q3_WHT:
      return 0.375 + (2.0 / 32.0);
    default:
      return 4.0;
  }
}

static void rebuild_decode_bandwidth_model(const LoadedModelState &state) {
  DecodeBandwidthModel model;
  model.source = "tensor-directory";
  model.output_tied = state.tensor_index.output_tied;

  const std::size_t embedding_index = state.tensor_index.token_embeddings;
  for (std::size_t index = 0; index < state.model.tensors.size(); index += 1) {
    const TensorInfo &tensor = state.model.tensors[index];
    model.total_tensor_bytes += tensor.byte_length;
    if (index == embedding_index) {
      model.embedding_tensor_bytes = tensor.byte_length;
    }
  }

  // A decode step reads every layer weight and the output projection once. The
  // token embedding table is a gather of a single row -- unless the output head
  // is tied to it, in which case the same matrix is also read in full to produce
  // logits and must stay in the count.
  std::uint64_t embedding_row_bytes = 0;
  if (embedding_index != k_missing_tensor_index && embedding_index < state.model.tensors.size()) {
    const TensorInfo &embedding = state.model.tensors[embedding_index];
    const std::uint64_t vocabulary = std::max<std::uint64_t>(1, state.model.token_vocabulary_size);
    embedding_row_bytes = embedding.byte_length / vocabulary;
  }
  model.weight_bytes_per_token = model.output_tied
    ? model.total_tensor_bytes
    : (model.total_tensor_bytes - model.embedding_tensor_bytes + embedding_row_bytes);

  const std::uint64_t layer_count = state.model.block_count.value_or(0);
  const std::uint64_t head_count = state.model.attention_head_count.value_or(0);
  const std::uint64_t kv_head_count = state.model.kv_head_count.value_or(head_count);
  const std::uint64_t hidden_size = state.model.embedding_length.value_or(0);
  const std::uint64_t key_dim = state.model.attention_key_length.value_or(
    head_count > 0 && hidden_size % head_count == 0 ? hidden_size / head_count : 0);
  const std::uint64_t value_dim = state.model.attention_value_length.value_or(key_dim);
  if (layer_count > 0 && kv_head_count > 0 && key_dim > 0 && value_dim > 0) {
    const double per_element = kv_bytes_per_element_estimate();
    const double elements = static_cast<double>(layer_count)
      * static_cast<double>(kv_head_count)
      * static_cast<double>(key_dim + value_dim);
    model.kv_bytes_per_token = static_cast<std::uint64_t>(elements * per_element);
  }

  model.ready = model.weight_bytes_per_token > 0;
  decode_bandwidth_model_storage() = model;
}

// The roofline denominator is measured per host and passed in, never hardcoded.
// Unset means the fraction is reported as absent -- a missing number is honest,
// a number divided by the wrong machine's bandwidth is not.
static double roofline_gigabytes_per_second() {
  static const double value = [] {
    const char *text = std::getenv("LIZARD_NATIVE_ROOFLINE_GBS");
    if (text == nullptr) {
      return 0.0;
    }
    const double parsed = std::strtod(text, nullptr);
    return parsed > 0.0 ? parsed : 0.0;
  }();
  return value;
}

struct ProcessResourceSnapshot {
  bool available = false;
  std::uint64_t cpu_100ns = 0;
  std::uint64_t working_set_bytes = 0;
  std::uint64_t private_bytes = 0;
};

static ProcessResourceSnapshot read_process_resource_snapshot() {
  ProcessResourceSnapshot snapshot;
#ifdef _WIN32
  FILETIME created {};
  FILETIME exited {};
  FILETIME kernel {};
  FILETIME user {};
  PROCESS_MEMORY_COUNTERS_EX memory {};
  memory.cb = sizeof(memory);
  const HANDLE process = GetCurrentProcess();
  if (GetProcessTimes(process, &created, &exited, &kernel, &user) == 0
    || GetProcessMemoryInfo(
      process,
      reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&memory),
      sizeof(memory)) == 0) {
    return snapshot;
  }
  ULARGE_INTEGER kernel_ticks {};
  kernel_ticks.LowPart = kernel.dwLowDateTime;
  kernel_ticks.HighPart = kernel.dwHighDateTime;
  ULARGE_INTEGER user_ticks {};
  user_ticks.LowPart = user.dwLowDateTime;
  user_ticks.HighPart = user.dwHighDateTime;
  snapshot.available = true;
  snapshot.cpu_100ns = kernel_ticks.QuadPart + user_ticks.QuadPart;
  snapshot.working_set_bytes = static_cast<std::uint64_t>(memory.WorkingSetSize);
  snapshot.private_bytes = static_cast<std::uint64_t>(memory.PrivateUsage);
#else
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return snapshot;
  }
  const auto timeval_to_100ns = [](const timeval &value) {
    return static_cast<std::uint64_t>(value.tv_sec) * 10000000ULL
      + static_cast<std::uint64_t>(value.tv_usec) * 10ULL;
  };
  snapshot.available = true;
  snapshot.cpu_100ns = timeval_to_100ns(usage.ru_utime) + timeval_to_100ns(usage.ru_stime);
  // ru_maxrss is KiB on Linux and bytes on macOS. Private bytes are not
  // available portably, so leave that field absent rather than guessing.
#ifdef __APPLE__
  snapshot.working_set_bytes = static_cast<std::uint64_t>(usage.ru_maxrss);
#else
  snapshot.working_set_bytes = static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
#endif
  return snapshot;
}

static void reset_decode_resource_metrics(SessionState &session) {
  session.resource_decode_elapsed_ms = 0;
  session.resource_decode_cpu_100ns = 0;
  session.resource_decode_token_count = 0;
  session.resource_working_set_bytes = 0;
  session.resource_private_bytes = 0;
  session.resource_peak_working_set_bytes = 0;
  session.resource_peak_private_bytes = 0;
  session.resource_metrics_available = false;
  session.resource_metrics_cohort = false;
}

static void append_decode_resource_metrics(
  SessionState &session,
  const ProcessResourceSnapshot &before,
  const ProcessResourceSnapshot &after,
  std::uint64_t elapsed_ms,
  std::uint64_t decoded_tokens,
  bool cohort) {
  if (!before.available || !after.available || after.cpu_100ns < before.cpu_100ns) {
    return;
  }
  session.resource_metrics_available = true;
  session.resource_metrics_cohort = session.resource_metrics_cohort || cohort;
  session.resource_decode_elapsed_ms += elapsed_ms;
  session.resource_decode_cpu_100ns += after.cpu_100ns - before.cpu_100ns;
  session.resource_decode_token_count += decoded_tokens;
  session.resource_working_set_bytes = after.working_set_bytes;
  session.resource_private_bytes = after.private_bytes;
  session.resource_peak_working_set_bytes = std::max(
    session.resource_peak_working_set_bytes,
    std::max(before.working_set_bytes, after.working_set_bytes));
  session.resource_peak_private_bytes = std::max(
    session.resource_peak_private_bytes,
    std::max(before.private_bytes, after.private_bytes));
}

static JsonValue build_decode_evidence_metrics_object(const SessionState &session) {
  if (!session.resource_metrics_available || session.resource_decode_elapsed_ms == 0) {
    return JsonValue();
  }
  const std::uint64_t logical_cpu_count = std::max<std::uint64_t>(
    1,
    static_cast<std::uint64_t>(std::thread::hardware_concurrency()));
  const double elapsed_seconds = static_cast<double>(session.resource_decode_elapsed_ms) / 1000.0;
  const double cpu_seconds = static_cast<double>(session.resource_decode_cpu_100ns) / 10000000.0;
  const double normalized_cpu_percent = elapsed_seconds > 0.0
    ? (cpu_seconds / elapsed_seconds) * 100.0 / static_cast<double>(logical_cpu_count)
    : 0.0;
  const double tokens_per_second = session.resource_decode_token_count > 0 && elapsed_seconds > 0.0
    ? static_cast<double>(session.resource_decode_token_count) / elapsed_seconds
    : 0.0;
  JsonValue tokens_per_full_cpu;
  if (tokens_per_second > 0.0 && normalized_cpu_percent > 0.0) {
    tokens_per_full_cpu = JsonValue(tokens_per_second / (normalized_cpu_percent / 100.0));
  }
  return JsonValue(JsonObject {
    {"schemaVersion", JsonValue(1)},
    {"measurementScope", JsonValue(session.resource_metrics_cohort ? "decode-cohort" : "decode")},
    {"sampleSource", JsonValue("native-worker-process")},
    {"tokensPerSecond", tokens_per_second > 0.0 ? JsonValue(tokens_per_second) : JsonValue()},
    {"normalizedCpuUtilizationPercent", JsonValue(normalized_cpu_percent)},
    {"cpuSeconds", JsonValue(cpu_seconds)},
    {"elapsedMs", JsonValue(session.resource_decode_elapsed_ms)},
    {"logicalCpuCount", JsonValue(logical_cpu_count)},
    {"workingSetBytes", JsonValue(session.resource_working_set_bytes)},
    {"privateBytes", session.resource_private_bytes > 0
      ? JsonValue(session.resource_private_bytes) : JsonValue()},
    {"peakWorkingSetBytes", JsonValue(session.resource_peak_working_set_bytes)},
    {"peakPrivateBytes", session.resource_peak_private_bytes > 0
      ? JsonValue(session.resource_peak_private_bytes) : JsonValue()},
    {"tokensPerSecondPerFullCpu", tokens_per_full_cpu},
  });
}

static JsonObject build_session_metrics_object(const SessionState &session) {
  std::uint64_t key_value_entries = 0;
  std::uint64_t kv_storage_bytes = 0;
  for (const SessionState::LayerKvCache &layer : session.layered_kv_cache) {
    key_value_entries += static_cast<std::uint64_t>(layer.keys.size() + layer.values.size());
    kv_storage_bytes += layer.keys.storage_bytes() + layer.values.storage_bytes();
  }
  JsonArray prefill_projection_metrics = build_prefill_projection_metrics_array(session);
  JsonArray decode_phase_metrics = build_decode_phase_metrics_array(session);
  const D3d12RuntimeTelemetry &gpu_telemetry = d3d12_runtime_telemetry();

  // EnhancementRun_04 Phase 0 — bandwidth view of the last decode window.
  const DecodeBandwidthModel &bandwidth = decode_bandwidth_model();
  JsonValue decode_tokens_per_second;
  JsonValue effective_weight_gbs;
  JsonValue roofline_fraction;
  if (session.last_decode_ms > 0 && session.last_decode_token_count > 0) {
    const double seconds = static_cast<double>(session.last_decode_ms) / 1000.0;
    const double tokens_per_second = static_cast<double>(session.last_decode_token_count) / seconds;
    decode_tokens_per_second = JsonValue(tokens_per_second);
    if (bandwidth.ready) {
      const double gigabytes_per_second =
        (static_cast<double>(bandwidth.weight_bytes_per_token) * tokens_per_second) / 1e9;
      effective_weight_gbs = JsonValue(gigabytes_per_second);
      const double roofline = roofline_gigabytes_per_second();
      if (roofline > 0.0) {
        roofline_fraction = JsonValue(gigabytes_per_second / roofline);
      }
    }
  }

  return {
    {"sessionId", JsonValue(session.session_id)},
    {"cancelled", JsonValue(session.cancelled)},
    {"nativeBackend", JsonValue(session.native_backend.empty() ? "cpu" : session.native_backend)},
    {"cpuBackend", build_cpu_backend_object()},
    {"gpuAdapter", build_gpu_adapter_object()},
    {"gpuMemory", build_gpu_memory_object()},
    {"prefillComplete", JsonValue(session.prefill_complete)},
    {"prefillBackend", JsonValue(session.prefill_backend.empty() ? "cpu" : session.prefill_backend)},
    {"decodeBackend", JsonValue(session.decode_backend.empty() ? "cpu" : session.decode_backend)},
    {"gpuDecodePolicy", JsonValue(session.gpu_decode_policy)},
    {"promptTokenCount", JsonValue(static_cast<std::uint64_t>(session.prompt_token_ids.size()))},
    {"cachedTokenCount", JsonValue(std::max<std::uint64_t>(
      session.cached_token_count,
      static_cast<std::uint64_t>(session.sequence_token_ids.size())))},
    {"generatedTokenCount", JsonValue(session.generated_token_count)},
    {"lastDecodeTokenCount", JsonValue(session.last_decode_token_count)},
    {"cpuWorkerCount", JsonValue(static_cast<std::uint64_t>(max_cpu_row_workers()))},
    {"kvCacheBytes", JsonValue(session.kv_cache_bytes)},
    {"kvCacheMode", JsonValue(kv_cache_mode_name())},
    {"kvCacheStorageBytes", JsonValue(kv_storage_bytes)},
    {"nativeSessionBytes", JsonValue(estimate_session_native_bytes(session))},
    {"layerCacheCount", JsonValue(static_cast<std::uint64_t>(session.layered_kv_cache.size()))},
    {"keyValueEntries", JsonValue(key_value_entries)},
    {"lastPrefillMs", JsonValue(session.last_prefill_ms)},
    {"lastDecodeMs", JsonValue(session.last_decode_ms)},
    {"nativeBackendDetail", JsonValue(session.native_backend_detail)},
    {"prefillBackendDetail", JsonValue(session.prefill_backend_detail)},
    {"decodeBackendDetail", JsonValue(session.decode_backend_detail)},
    {"decodeQkvFuseBlocked", JsonValue(session.decode_qkv_fuse_blocked)},
    {"decodeQkvFuseDetail", JsonValue(session.decode_qkv_fuse_detail)},
    {"decodeFfnFuseBlocked", JsonValue(session.decode_ffn_fuse_blocked)},
    {"decodeFfnFuseDetail", JsonValue(session.decode_ffn_fuse_detail)},
    {"decodeLayerTailFuseBlocked", JsonValue(session.decode_layer_tail_fuse_blocked)},
    {"decodeLayerTailFuseDetail", JsonValue(session.decode_layer_tail_fuse_detail)},
    {"prefillProjectionMetrics", JsonValue(prefill_projection_metrics)},
    {"decodePhaseMetrics", JsonValue(decode_phase_metrics)},
    {"evidenceMetrics", build_decode_evidence_metrics_object(session)},
    {"gpuSubmissionCount", JsonValue(gpu_telemetry.submission_count)},
    {"gpuFenceWaitCount", JsonValue(gpu_telemetry.fence_wait_count)},
    {"gpuTokenGraphSubmissionCount", JsonValue(gpu_telemetry.token_graph_submission_count)},
    {"gpuWeightCacheHits", JsonValue(gpu_telemetry.weight_cache_hits)},
    {"gpuWeightCacheMisses", JsonValue(gpu_telemetry.weight_cache_misses)},
    {"gpuWeightCacheEvictions", JsonValue(gpu_telemetry.weight_cache_evictions)},
    {"gpuWeightUploadBytes", JsonValue(gpu_telemetry.weight_upload_bytes)},
    {"gpuInputUploadBytes", JsonValue(gpu_telemetry.input_upload_bytes)},
    {"gpuOutputReadbackBytes", JsonValue(gpu_telemetry.output_readback_bytes)},
    {"decodeTokensPerSecond", decode_tokens_per_second},
    {"weightBytesPerToken", JsonValue(bandwidth.weight_bytes_per_token)},
    {"kvBytesPerToken", JsonValue(bandwidth.kv_bytes_per_token)},
    {"effectiveWeightGBs", effective_weight_gbs},
    {"rooflineGBs", roofline_gigabytes_per_second() > 0.0
      ? JsonValue(roofline_gigabytes_per_second())
      : JsonValue()},
    {"rooflineFraction", roofline_fraction},
    {"decodeBandwidthSource", JsonValue(bandwidth.ready ? bandwidth.source : std::string("unavailable"))},
    {"outputTied", JsonValue(bandwidth.output_tied)},
  };
}

static JsonObject build_tensor_index_summary(const LoadedModelState &state) {
  JsonArray missing_shared;
  for (const std::string &name : state.tensor_index.missing_shared) {
    missing_shared.push_back(JsonValue(name));
  }

  JsonObject summary = {
    {"ready", JsonValue(state.tensor_index.ready)},
    {"layerCount", JsonValue(static_cast<std::uint64_t>(state.tensor_index.layers.size()))},
    {"zeroLayerExecutable", JsonValue(zero_layer_graph_executable(state))},
    {"llamaGraphExecutable", JsonValue(llama_graph_executable(state))},
    {"gemmaGraphExecutable", JsonValue(layered_gemma_graph_executable(state))},
    {"graphExecutable", JsonValue(native_graph_executable(state))},
    {"outputTied", JsonValue(state.tensor_index.output_tied)},
    {"ropeFreqsPresent", JsonValue(state.tensor_index.rope_freqs != k_missing_tensor_index)},
    {"cachedVectorTensorCount", JsonValue(static_cast<std::uint64_t>(state.vector_tensor_cache.size()))},
    {"cachedGemmaLayerConfigCount", JsonValue(static_cast<std::uint64_t>(state.gemma_layer_configs.size()))},
    {"gemmaLayerConfigsReady", JsonValue(state.gemma_layer_configs_ready)},
    {"missingShared", JsonValue(missing_shared)},
    {"missingLayers", JsonValue(state.tensor_index.missing_layers)},
    {"supportedDequantTypes", JsonValue(state.tensor_index.supported_dequant_types)},
    {"tokenizer", JsonValue(JsonObject {
      {"tokenizer", JsonValue(state.model.tokenizer.empty() ? "unknown" : state.model.tokenizer)},
      {"tokenVocabularySize", JsonValue(state.model.token_vocabulary_size)},
      {"tokenMergeCount", JsonValue(state.model.token_merge_count)},
      {"chatTemplatePresent", JsonValue(!state.model.chat_template.empty())},
      {"specialTokensResolved", JsonValue(!state.model.special_token_texts.empty())},
      {"exactPromptTokenizationAvailable", JsonValue(!state.sorted_token_indices.empty())},
      {"gpt2PromptTokenizationAvailable", JsonValue(
        to_lower(state.model.tokenizer) == "gpt2"
        && !state.token_to_id.empty()
        && !state.merge_ranks.empty())},
      {"ready", JsonValue(state.model.token_vocabulary_size > 0 && !state.model.special_token_texts.empty())},
    })},
  };

  JsonArray sampled_tensors;
  const std::vector<std::pair<std::string, std::size_t>> sample_candidates = {
    {"tokenEmbeddings", state.tensor_index.token_embeddings},
    {"outputNorm", state.tensor_index.output_norm},
    {"output", state.tensor_index.output},
  };
  for (const auto &[label, index] : sample_candidates) {
    if (index == k_missing_tensor_index) {
      continue;
    }
    const TensorInfo &tensor = tensor_at(state, index);
    if (!(tensor.ggml_type == 0 || tensor.ggml_type == 1 || tensor.ggml_type == 8 || tensor.ggml_type == 12 || tensor.ggml_type == 13 || tensor.ggml_type == 14 || tensor.ggml_type == 30)) {
      continue;
    }
    std::vector<float> values;
    dequantize_tensor_range_into_f32(
      state,
      tensor,
      0,
      std::min<std::uint64_t>(8, tensor.element_count),
      values);
    JsonArray preview;
    for (std::size_t item = 0; item < values.size(); item += 1) {
      preview.push_back(JsonValue(static_cast<double>(values[item])));
    }
    sampled_tensors.push_back(JsonValue(JsonObject {
      {"label", JsonValue(label)},
      {"name", JsonValue(tensor.name)},
      {"ggmlTypeName", JsonValue(tensor.ggml_type_name.empty() ? "unknown" : tensor.ggml_type_name)},
      {"preview", JsonValue(preview)},
    }));
  }
  summary.emplace("sampledTensors", JsonValue(sampled_tensors));
  return summary;
}

static JsonValue success_response(const JsonValue &id, JsonObject result) {
  return JsonValue(JsonObject {
    {"id", id},
    {"ok", JsonValue(true)},
    {"result", JsonValue(std::move(result))},
  });
}

static JsonValue error_response(const JsonValue &id, const WorkerError &error) {
  JsonObject error_object = {
    {"code", JsonValue(error.code())},
    {"message", JsonValue(std::string(error.what()))},
    {"details", error.details()},
  };
  return JsonValue(JsonObject {
    {"id", id},
    {"ok", JsonValue(false)},
    {"error", JsonValue(error_object)},
  });
}

static std::string compatibility_failure_reason(const JsonObject &compatibility) {
  const JsonObject *architecture = find_object_field(compatibility, "architecture");
  const JsonObject *quantization = find_object_field(compatibility, "quantization");
  if (architecture != nullptr && json_string_or_default(*architecture, "status") != "supported") {
    return json_string_or_default(*architecture, "reason", "Unsupported architecture.");
  }
  if (quantization != nullptr && json_string_or_default(*quantization, "status") != "supported") {
    return json_string_or_default(*quantization, "reason", "Unsupported quantization.");
  }
  return "";
}

// The adapter the D3D12 context actually selected (Phase 1 Finding 1):
// stable LUID identity + native UMA facts + the selection reason, so
// telemetry can prove the plan and the execution used the same GPU. Null
// until the D3D12 context has initialized (or when it failed to).
static JsonValue build_gpu_adapter_object() {
  // Observe only: never trigger D3D12 initialization from telemetry. A
  // CPU-only load would otherwise pay full device + PSO-compile cost just
  // to report "no adapter" (hung the BF16 zero-layer worker test for 50 s).
  const D3d12BatchMatmulCapabilities *caps_ptr = d3d12_batch_matmul_capabilities_if_initialized();
  if (caps_ptr == nullptr) {
    return JsonValue();
  }
  const D3d12BatchMatmulCapabilities &caps = *caps_ptr;
  if (!caps.adapter_identity_valid) {
    return JsonValue();
  }
  JsonObject adapter = {
    {"name", JsonValue(caps.adapter_name)},
    {"luidHigh", JsonValue(static_cast<double>(caps.adapter_luid_high))},
    {"luidLow", JsonValue(static_cast<double>(caps.adapter_luid_low))},
    {"vendorId", JsonValue(static_cast<std::uint64_t>(caps.adapter_vendor_id))},
    {"deviceId", JsonValue(static_cast<std::uint64_t>(caps.adapter_device_id))},
    {"uma", JsonValue(caps.adapter_uma)},
    {"cacheCoherentUma", JsonValue(caps.adapter_cache_coherent_uma)},
    {"selectionReason", JsonValue(caps.adapter_selection_reason)},
  };
  return JsonValue(std::move(adapter));
}

// Live GPU memory accounting (Phase 1 Finding 2): DXGI budget/usage for
// both segment groups, the reserves, and the requested vs effective
// resident-weight budget. Null until the GPU lane has initialized;
// confidence "static-fallback" when DXGI queries are unavailable.
static JsonValue build_gpu_memory_object() {
  if (d3d12_batch_matmul_capabilities_if_initialized() == nullptr) {
    return JsonValue();
  }
  const D3d12GpuMemoryReport report = d3d12_gpu_memory_report();
  JsonObject memory = {
    {"live", JsonValue(report.live)},
    {"confidence", JsonValue(report.confidence)},
    {"localBudgetBytes", JsonValue(report.local_budget_bytes)},
    {"localUsageBytes", JsonValue(report.local_usage_bytes)},
    {"nonLocalBudgetBytes", JsonValue(report.non_local_budget_bytes)},
    {"nonLocalUsageBytes", JsonValue(report.non_local_usage_bytes)},
    {"reserveBytes", JsonValue(report.reserve_bytes)},
    {"requestedWeightBudgetBytes", JsonValue(report.requested_weight_budget_bytes)},
    {"effectiveWeightBudgetBytes", JsonValue(report.effective_weight_budget_bytes)},
    {"residentWeightBytes", JsonValue(report.resident_weight_bytes)},
  };
  return JsonValue(std::move(memory));
}

static JsonValue build_cpu_backend_object() {
  const std::string selection_error = ggml_cpu_backend_selection_error();
  JsonObject backend = {
    {"effective", JsonValue(ggml_cpu_backend_effective_name())},
    {"ggmlCompiled", JsonValue(ggml_cpu_backend_compiled())},
    {"ggmlSelected", JsonValue(ggml_cpu_backend_selected())},
    {"ggmlCommit", JsonValue(k_lizard_ggml_commit)},
    {"ggmlMatvecCalls", JsonValue(ggml_cpu_backend_matvec_calls())},
    {"ggmlMatvecFailures", JsonValue(ggml_cpu_backend_matvec_failures())},
    {"ggmlPlanHits", JsonValue(ggml_cpu_backend_plan_hits())},
    {"ggmlPlanMisses", JsonValue(ggml_cpu_backend_plan_misses())},
    {"ggmlMatvecNanoseconds", JsonValue(ggml_cpu_backend_matvec_nanoseconds())},
    {"ggmlWeightBytes", JsonValue(ggml_cpu_backend_weight_bytes())},
    {"ggmlRepackBudgetBytes", JsonValue(ggml_cpu_backend_repack_budget_bytes())},
    {"ggmlRepackedBytes", JsonValue(ggml_cpu_backend_repacked_bytes())},
    {"ggmlRepackedTensors", JsonValue(ggml_cpu_backend_repacked_tensors())},
    {"ggmlRepackFailures", JsonValue(ggml_cpu_backend_repack_failures())},
    {"ggmlGroupedGraphsEnabled", JsonValue(ggml_cpu_backend_grouped_graphs_enabled())},
    {"ggmlGroupedGraphCalls", JsonValue(ggml_cpu_backend_grouped_graph_calls())},
    {"ggmlGroupedGraphOperations", JsonValue(ggml_cpu_backend_grouped_graph_operations())},
    {"ggmlFusedFfnEnabled", JsonValue(ggml_cpu_backend_fused_ffn_enabled())},
    {"ggmlFusedFfnCalls", JsonValue(ggml_cpu_backend_fused_ffn_calls())},
    {"selectionValid", JsonValue(selection_error.empty())},
    {"selectionError", JsonValue(selection_error)},
  };
  return JsonValue(std::move(backend));
}

static JsonObject build_handshake_result(const WorkerState &state) {
  JsonArray methods = {
    JsonValue("handshake"),
    JsonValue("probe"),
    JsonValue("load_model"),
    JsonValue("unload_model"),
    JsonValue("session_open"),
    JsonValue("session_clone"),
    JsonValue("session_close"),
    JsonValue("prefill"),
    JsonValue("decode"),
    JsonValue("generate"),
    JsonValue("decode_batch"),
    JsonValue("cancel"),
    JsonValue("metrics"),
    JsonValue("tensor_benchmark"),
    JsonValue("shutdown"),
  };
  JsonArray architectures = {
    JsonValue("llama"),
    JsonValue("gemma"),
    JsonValue("qwen2"),
    JsonValue("qwen3"),
    JsonValue("mistral"),
    JsonValue("mixtral"),
    JsonValue("phi"),
    JsonValue("phi3"),
    JsonValue("qwen2moe"),
    JsonValue("qwen3moe"),
    JsonValue("deepseek"),
    JsonValue("glm4"),
    JsonValue("starcoder2"),
  };
  JsonArray quantizations = {
    JsonValue("f32"),
    JsonValue("f16"),
    JsonValue("bf16"),
    JsonValue("q8_0"),
    JsonValue("q4_0"),
    JsonValue("q2_k"),
    JsonValue("q3_k_m"),
    JsonValue("q4_k_m"),
    JsonValue("q5_k_m"),
    JsonValue("q6_k"),
    JsonValue("iq4_xs"),
    JsonValue("tq1_0"),
    JsonValue("tq2_0"),
  };
  JsonArray load_modes = {
    JsonValue("mmap"),
    JsonValue("no-mmap"),
    JsonValue("mlock"),
  };
  JsonArray hybrid_backends;
#ifdef _WIN32
  // Handshake must stay cheap. Actual adapter/device validation is deferred
  // to probe or to loading a model with GPU-sized projection tensors.
  hybrid_backends.push_back(JsonValue("directx-hybrid"));
#endif

  return {
    {"worker", JsonValue("lizard-native-runner-cpp")},
#ifdef LIZARD_NATIVE_HOST_OPTIMIZED
    {"cpuBuildProfile", JsonValue("host-optimized")},
#elif defined(LIZARD_NATIVE_AVX512)
    {"cpuBuildProfile", JsonValue("avx512-x86-64")},
#elif defined(LIZARD_NATIVE_AVX2)
    {"cpuBuildProfile", JsonValue("avx2-x86-64")},
#else
    {"cpuBuildProfile", JsonValue("portable-x86-64")},
#endif
    {"rpcVersion", JsonValue(static_cast<std::uint64_t>(state.rpc_version))},
    {"runtimeProvider", JsonValue("lizard-native")},
    {"cpuBackend", build_cpu_backend_object()},
    {"methods", JsonValue(methods)},
    {"architectures", JsonValue(architectures)},
    {"quantizations", JsonValue(quantizations)},
    {"loadModes", JsonValue(load_modes)},
    {"hybridBackends", JsonValue(hybrid_backends)},
  };
}

static JsonObject handle_probe() {
  const D3d12BatchMatmulCapabilities &gpu_capabilities = d3d12_batch_matmul_capabilities();
  JsonArray hybrid_backends;
  if (gpu_capabilities.supported) {
    hybrid_backends.push_back(JsonValue(gpu_capabilities.backend_name));
  }
  return {
    {"runtimeProvider", JsonValue("lizard-native")},
    {"cpuBackend", build_cpu_backend_object()},
#ifdef LIZARD_NATIVE_HOST_OPTIMIZED
    {"cpuBuildProfile", JsonValue("host-optimized")},
#elif defined(LIZARD_NATIVE_AVX512)
    {"cpuBuildProfile", JsonValue("avx512-x86-64")},
#elif defined(LIZARD_NATIVE_AVX2)
    {"cpuBuildProfile", JsonValue("avx2-x86-64")},
#else
    {"cpuBuildProfile", JsonValue("portable-x86-64")},
#endif
    {"backend", JsonValue(gpu_capabilities.supported ? gpu_capabilities.backend_name : "cpu")},
    {"experimental", JsonValue(true)},
    {"executionReady", JsonValue(false)},
    {"backendReason", JsonValue(gpu_capabilities.reason)},
    {"hybridBackends", JsonValue(hybrid_backends)},
    {"architectures", JsonValue(JsonArray {
      JsonValue("llama"),
      JsonValue("gemma"),
      JsonValue("qwen2"),
      JsonValue("qwen3"),
      JsonValue("mistral"),
      JsonValue("mixtral"),
      JsonValue("phi"),
      JsonValue("phi3"),
      JsonValue("qwen2moe"),
      JsonValue("qwen3moe"),
      JsonValue("deepseek"),
      JsonValue("glm4"),
      JsonValue("starcoder2"),
    })},
    {"quantizations", JsonValue(JsonArray {
      JsonValue("f32"),
      JsonValue("f16"),
      JsonValue("bf16"),
      JsonValue("q8_0"),
      JsonValue("q4_0"),
      JsonValue("q2_k"),
      JsonValue("q3_k_m"),
      JsonValue("q4_k_m"),
      JsonValue("q5_k_m"),
      JsonValue("q6_k"),
      JsonValue("iq4_xs"),
      JsonValue("tq1_0"),
      JsonValue("tq2_0"),
    })},
  };
}

static JsonObject handle_load_model(WorkerState &state, const JsonObject &params) {
  const JsonObject *model_artifact = find_object_field(params, "modelArtifact");
  if (model_artifact == nullptr) {
    throw WorkerError("NATIVE_WORKER_ERROR", "load_model requires modelArtifact.");
  }
  const std::string model_path = json_string_or_default(*model_artifact, "modelPath");
  if (model_path.empty()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "load_model requires modelArtifact.modelPath.");
  }
  if (!ends_with_ci(model_path, ".gguf")) {
    throw WorkerError(
      "NATIVE_MODEL_UNSUPPORTED",
      "The compiled native worker currently accepts only GGUF files. Prototype JSON models still use the JS worker path.",
      JsonObject {{"modelPath", JsonValue(model_path)}});
  }

  const JsonObject *profile = find_object_field(params, "profile");
  const bool use_mmap = profile == nullptr ? true : json_bool_or_default(*profile, "mmap", true);
  const bool pin_memory = profile != nullptr && json_bool_or_default(*profile, "mlock", false);
  const auto started_at = std::chrono::steady_clock::now();

  auto file_view = std::make_unique<FileView>();
  file_view->open(std::filesystem::path(model_path), use_mmap, pin_memory);
  ParsedGgufModel model = parse_gguf(*file_view, std::filesystem::path(model_path));
  JsonObject compatibility = build_compatibility_object(model);

  // Drop stale device weight copies so a new mapping cannot alias cached keys.
  d3d12_reset_resident_weights();
  ggml_cpu_backend_reset();
  state.loaded_model = std::make_unique<LoadedModelState>();
  state.loaded_model->model = model;
  state.loaded_model->compatibility = compatibility;
  state.loaded_model->tensor_index = build_llama_tensor_index(state.loaded_model->model);
  // Cheap: a name lookup per tensor, no weights touched. A text-only model
  // simply comes back not-ready.
  state.loaded_model->vision_index = build_vision_tensor_index(state.loaded_model->model);
  state.loaded_model->sorted_token_indices = sort_token_indices_by_match_priority(state.loaded_model->model.token_vocabulary);
  for (std::size_t index = 0; index < state.loaded_model->model.token_vocabulary.size(); index += 1) {
    const std::string &token = state.loaded_model->model.token_vocabulary[index];
    if (!token.empty() && state.loaded_model->token_to_id.find(token) == state.loaded_model->token_to_id.end()) {
      state.loaded_model->token_to_id.emplace(token, index);
    }
  }
  for (std::size_t index = 0; index < state.loaded_model->model.token_merges.size(); index += 1) {
    const std::string &merge = state.loaded_model->model.token_merges[index];
    if (!merge.empty() && state.loaded_model->merge_ranks.find(merge) == state.loaded_model->merge_ranks.end()) {
      state.loaded_model->merge_ranks.emplace(merge, index);
    }
  }
  state.loaded_model->file_view = std::move(file_view);
  build_vector_tensor_cache(*state.loaded_model);
  build_gemma_layer_execution_config_cache(*state.loaded_model);
  build_llama_layer_shape_cache(*state.loaded_model);
  // Optional second file: an mmproj carrying the vision tower.
  //
  // Loaded as a model state in its own right rather than merged into the first
  // one, because a merge would need tensor indices from two files to share a
  // numbering - and every accessor in this runner resolves an index against one
  // model. Keeping them separate means nothing else has to learn that a second
  // file exists.
  const std::string projector_path = json_string_or_default(*model_artifact, "visionProjectorPath");
  if (!projector_path.empty()) {
    if (!ends_with_ci(projector_path, ".gguf")) {
      throw WorkerError("NATIVE_MODEL_UNSUPPORTED", "The vision projector must be a GGUF file.");
    }
    auto projector_view = std::make_unique<FileView>();
    projector_view->open(std::filesystem::path(projector_path), use_mmap, pin_memory);
    ParsedGgufModel projector = parse_gguf(*projector_view, std::filesystem::path(projector_path));
    auto projector_state = std::make_shared<LoadedModelState>();
    projector_state->model = projector;
    projector_state->vision_index = build_vision_tensor_index(projector_state->model);
    projector_state->file_view = std::move(projector_view);
    build_vector_tensor_cache(*projector_state);
    if (!projector_state->vision_index.ready) {
      // Refuse rather than carry on text-only: the caller asked for a vision
      // projector by path, so a file that is not one is a mistake worth naming.
      std::string detail;
      for (const std::string &missing : projector_state->vision_index.missing) {
        detail += (detail.empty() ? "" : ", ") + missing;
      }
      throw WorkerError(
        "NATIVE_VISION_PROJECTOR_INVALID",
        "That file does not contain a usable vision tower" + (detail.empty() ? "." : " (" + detail + ")."));
    }
    // The projector's own language width must match the model it is feeding, or
    // the spliced rows are the wrong shape and every later error is downstream
    // of this one.
    const std::uint64_t model_width = state.loaded_model->model.embedding_length.value_or(0);
    const std::uint64_t projector_width = projector_state->vision_index.language_hidden_size;
    if (model_width != 0 && projector_width != 0 && model_width != projector_width) {
      throw WorkerError(
        "NATIVE_VISION_PROJECTOR_MISMATCH",
        "This projector emits " + std::to_string(projector_width) + "-wide rows but the model's embedding width is "
          + std::to_string(model_width) + "; they are not a matching pair.");
    }
    state.loaded_model->projector_model = std::move(projector_state);
  }

  state.loaded_model->gpu_projection_candidate =
    model_has_gpu_projection_candidate(*state.loaded_model);
  state.sessions.clear();
  const bool execution_ready = native_graph_executable(*state.loaded_model);

  // Pre-stage projection weights on the GPU so the first request does not
  // pay the cold staging cost. Tensors above the budget or per-tensor cap
  // stage lazily on first use instead. Disable with
  // LIZARD_NATIVE_GPU_PRESTAGE=0.
  std::uint64_t gpu_prestage_ms = 0;
  if (execution_ready
    && state.loaded_model->gpu_projection_candidate
    && d3d12_batch_matmul_capabilities().supported) {
    // Re-read the adapter's live budget before committing multi-GB staging:
    // memory pressure may have changed since the context initialized
    // (Finding 2 recheck point "before model prestaging").
    d3d12_refresh_gpu_memory_budget();
    const char *prestage_env = std::getenv("LIZARD_NATIVE_GPU_PRESTAGE");
    if (prestage_env == nullptr || std::string(prestage_env) != "0") {
      const auto prestage_started_at = std::chrono::steady_clock::now();
      // Pre-stage fills the cache up to the budget and stops; it must not evict
      // what it just staged. Decode-time staging (outside this window) evicts.
      d3d12_set_prestaging(true);
      for (const TensorInfo &tensor : state.loaded_model->model.tensors) {
        if (tensor.dimensions.size() != 2) {
          continue;
        }
        const std::uint64_t row_width = tensor.dimensions[0];
        const std::uint64_t row_count = tensor.dimensions[1];
        if (row_width == 0
          || row_count == 0
          || row_width * row_count != tensor.element_count
          || tensor.element_count < 1048576
          || row_count > std::numeric_limits<std::uint32_t>::max()) {
          continue;
        }
        D3d12WeightFormat gpu_format = D3d12WeightFormat::F32;
        if (!d3d12_weight_format_supported(tensor.ggml_type, row_width, &gpu_format)) {
          continue;
        }
        const std::uint8_t *weight_data = tensor_data_ptr(*state.loaded_model, tensor);
        D3d12ResidentWeightRef op;
        op.weight_key = weight_data;
        op.format = gpu_format;
        op.weight_data = weight_data;
        op.weight_bytes = static_cast<std::size_t>(tensor.byte_length);
        op.row_count = static_cast<std::uint32_t>(row_count);
        op.row_width = static_cast<std::uint32_t>(row_width);
        d3d12_stage_resident_weight(op);
      }
      d3d12_set_prestaging(false);
      d3d12_shrink_staging_buffers();
      gpu_prestage_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - prestage_started_at).count());
    }
  }

  // EnhancementRun_04 Phase 0: the weight-byte count depends only on the tensor
  // directory, so it is computed once here rather than per decode step.
  rebuild_decode_bandwidth_model(*state.loaded_model);

  const auto finished_at = std::chrono::steady_clock::now();
  state.load_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count()));
  state.loaded_model->load_ms = state.load_ms;
  JsonArray hybrid_backends;
  if (state.loaded_model->gpu_projection_candidate
    && d3d12_batch_matmul_capabilities().supported) {
    hybrid_backends.push_back(JsonValue(d3d12_batch_matmul_capabilities().backend_name));
  }

  JsonObject result = {
    {"runtimeProvider", JsonValue("lizard-native")},
    {"nativeBackend", JsonValue(ggml_cpu_backend_effective_name())},
    {"cpuBackend", build_cpu_backend_object()},
    {"nativeModelStatus", JsonValue(native_model_status(state.loaded_model.get()))},
    {"executionReady", JsonValue(execution_ready)},
    {"loadMs", JsonValue(state.load_ms)},
    {"model", JsonValue(build_model_object(state.loaded_model->model))},
    {"compatibility", JsonValue(state.loaded_model->compatibility)},
    {"executorState", JsonValue(build_tensor_index_summary(*state.loaded_model))},
    {"hybridBackends", JsonValue(hybrid_backends)},
    {"gpuResidentWeightBytes", JsonValue(
      state.loaded_model->gpu_projection_candidate ? d3d12_resident_weight_bytes() : 0)},
    {"gpuPrestageMs", JsonValue(gpu_prestage_ms)},
    {"gpuAdapter", build_gpu_adapter_object()},
    {"gpuMemory", build_gpu_memory_object()},
  };
  return result;
}

static JsonObject handle_unload_model(WorkerState &state) {
  d3d12_reset_resident_weights();
  ggml_cpu_backend_reset();
  reset_decode_bandwidth_model();
  state.loaded_model.reset();
  state.load_ms = 0;
  state.sessions.clear();
  state.last_metrics = JsonValue();
  return {
    {"nativeModelStatus", JsonValue("unloaded")},
  };
}

static JsonObject handle_session_open(WorkerState &state, const JsonObject &params) {
  const std::string session_id = json_string_or_default(params, "sessionId");
  if (session_id.empty()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "session_open requires sessionId.");
  }
  SessionState session;
  session.session_id = session_id;
  static std::atomic<std::uint64_t> next_decode_equivalence_id {1};
  session.decode_equivalence_id =
    next_decode_equivalence_id.fetch_add(1, std::memory_order_relaxed);
  state.sessions[session_id] = session;
  return {
    {"sessionId", JsonValue(session_id)},
    {"nativeModelStatus", JsonValue(native_model_status(state.loaded_model.get()))},
    {"sessionMetrics", JsonValue(build_session_metrics_object(state.sessions[session_id]))},
  };
}

// Copies a completed prefill into a new session. This is intentionally a
// snapshot copy rather than shared mutable KV: each decode can append and
// truncate independently, while an exact repeated prompt pays the model
// traversal once. GPU-side KV and recorded graph descriptors are invalidated;
// the copied CPU KV is authoritative and will seed a fresh device cache if the
// target later takes a GPU lane.
static JsonObject handle_session_clone(WorkerState &state, const JsonObject &params) {
  const std::string source_session_id = json_string_or_default(params, "sourceSessionId");
  const std::string session_id = json_string_or_default(params, "sessionId");
  if (source_session_id.empty() || session_id.empty()) {
    throw WorkerError(
      "NATIVE_WORKER_ERROR",
      "session_clone requires sourceSessionId and sessionId.");
  }
  if (source_session_id == session_id) {
    throw WorkerError("NATIVE_WORKER_ERROR", "session_clone source and target must differ.");
  }
  const auto source = state.sessions.find(source_session_id);
  if (source == state.sessions.end()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "session_clone received an unknown source session.");
  }
  if (!source->second.prefill_complete || source->second.last_logits.empty()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "session_clone requires a completed native prefill.");
  }

  const auto started_at = std::chrono::steady_clock::now();
  SessionState cloned = source->second;
  cloned.session_id = session_id;
  cloned.cancelled = false;
  cloned.max_tokens = static_cast<std::uint64_t>(
    json_number_or_default(params, "maxTokens", static_cast<double>(cloned.max_tokens)));
  if (find_object_value(params, "stopSequences") != nullptr) {
    cloned.stop_sequences = json_array_to_strings(find_object_value(params, "stopSequences"));
  }
  cloned.batch_stop_scan_tail.clear();
  cloned.generated_token_count = 0;
  cloned.last_decode_token_count = 0;
  cloned.last_decode_ms = 0;
  cloned.decode_phase_metrics.clear();
  cloned.sampling = SessionState::SamplingState {};
  cloned.stream_tokens = false;
  cloned.gpu_kv_cache_generation = next_gpu_kv_cache_generation_value();
  cloned.gpu_kv_max_positions = 0;
  cloned.gpu_kv_cache_active.assign(cloned.layered_kv_cache.size(), false);
  cloned.gpu_token_graph_layers.clear();
  cloned.gpu_token_graph_descriptors_ready = false;
  cloned.gpu_token_graph_descriptor_generation = 0;
  cloned.last_gpu_argmax_token = -1;

  const std::uint64_t clone_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started_at).count()));
  cloned.last_prefill_ms = clone_ms;
  cloned.prefill_backend_detail = "Exact prompt prefill restored from an isolated session snapshot.";
  state.sessions[session_id] = std::move(cloned);
  SessionState &target = state.sessions[session_id];

  JsonArray prompt_token_ids;
  for (std::uint64_t token_id : target.prompt_token_ids) {
    prompt_token_ids.push_back(JsonValue(token_id));
  }
  return {
    {"sessionId", JsonValue(session_id)},
    {"sourceSessionId", JsonValue(source_session_id)},
    {"nativeBackend", JsonValue(target.native_backend)},
    {"prefillMs", JsonValue(clone_ms)},
    {"kvCacheBytes", JsonValue(target.kv_cache_bytes)},
    {"promptTokenIds", JsonValue(prompt_token_ids)},
    {"promptTokenCount", JsonValue(static_cast<std::uint64_t>(target.prompt_token_ids.size()))},
    {"cachedTokenCount", JsonValue(target.cached_token_count)},
    {"promptTokenMode", JsonValue(target.prompt_token_mode.empty() ? "unavailable" : target.prompt_token_mode)},
    {"kvPrefixReusedTokens", JsonValue(static_cast<std::uint64_t>(target.prompt_token_ids.size()))},
    {"prefillSnapshotHit", JsonValue(true)},
    {"sessionMetrics", JsonValue(build_session_metrics_object(target))},
  };
}

static JsonObject handle_session_close(WorkerState &state, const JsonObject &params) {
  const std::string session_id = json_string_or_default(params, "sessionId");
  if (session_id.empty()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "session_close requires sessionId.");
  }
  const auto iterator = state.sessions.find(session_id);
  if (iterator == state.sessions.end()) {
    return {
      {"sessionId", JsonValue(session_id)},
      {"closed", JsonValue(false)},
    };
  }
  const JsonObject metrics = build_session_metrics_object(iterator->second);
  state.sessions.erase(iterator);
  return {
    {"sessionId", JsonValue(session_id)},
    {"closed", JsonValue(true)},
    {"sessionMetrics", JsonValue(metrics)},
  };
}

static JsonObject handle_prefill(WorkerState &state, const JsonObject &params) {
  const std::string session_id = json_string_or_default(params, "sessionId");
  auto iterator = state.sessions.find(session_id);
  if (iterator == state.sessions.end()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Unknown native worker session.");
  }
  const auto prefill_started_at = std::chrono::steady_clock::now();
  reset_decode_resource_metrics(iterator->second);
  static std::atomic<std::uint64_t> next_prefill_equivalence_id {1ull << 32u};
  iterator->second.decode_equivalence_id =
    next_prefill_equivalence_id.fetch_add(1, std::memory_order_relaxed);
  iterator->second.cancelled = false;
  const JsonObject *prompt_artifact = find_object_field(params, "promptArtifact");
  if (prompt_artifact != nullptr) {
    iterator->second.prompt = json_string_or_default(*prompt_artifact, "renderedPrompt");
    iterator->second.prompt_token_ids = json_array_to_u64s(find_object_value(*prompt_artifact, "tokenIds"));
    iterator->second.prompt_token_mode = json_string_or_default(*prompt_artifact, "tokenIdMode");
  } else {
    iterator->second.prompt = json_string_or_default(params, "prompt");
  }
  if (iterator->second.prompt_token_ids.empty() && state.loaded_model != nullptr) {
    const auto exact_tokens = tokenize_exact_vocab(
      iterator->second.prompt,
      state.loaded_model->model.token_vocabulary,
      state.loaded_model->sorted_token_indices);
    if (exact_tokens.has_value()) {
      iterator->second.prompt_token_ids = std::move(exact_tokens.value());
      prepend_bos_if_missing(iterator->second.prompt_token_ids, state.loaded_model->model);
      iterator->second.prompt_token_mode = "exact-vocab";
    } else if (to_lower(state.loaded_model->model.tokenizer) == "gpt2") {
      const auto gpt2_tokens = tokenize_gpt2_bpe_ascii(
        iterator->second.prompt,
        state.loaded_model->model,
        state.loaded_model->token_to_id,
        state.loaded_model->merge_ranks);
      if (gpt2_tokens.has_value()) {
        iterator->second.prompt_token_ids = std::move(gpt2_tokens.value());
        iterator->second.prompt_token_mode = "gpt2-bpe";
      }
    }
  }
  // Capture the pre-prefill sequence before the bookkeeping below overwrites
  // it: prefix reuse compares it against the incoming prompt token ids.
  const std::vector<std::uint64_t> previous_sequence_token_ids =
    std::move(iterator->second.sequence_token_ids);
  const std::uint64_t previous_cached_token_count = iterator->second.cached_token_count;
  iterator->second.sequence_token_ids = iterator->second.prompt_token_ids;
  iterator->second.generated_token_count = 0;
  iterator->second.last_prefill_ms = 0;
  iterator->second.last_decode_ms = 0;
  iterator->second.last_hidden_state.clear();
  iterator->second.last_logits.clear();
  iterator->second.cached_token_count = 0;
  iterator->second.prefill_complete = false;
  iterator->second.native_backend = ggml_cpu_backend_effective_name();
  iterator->second.native_backend_detail.clear();
  iterator->second.prefill_backend = ggml_cpu_backend_effective_name();
  iterator->second.prefill_backend_detail.clear();
  iterator->second.decode_backend = ggml_cpu_backend_effective_name();
  iterator->second.decode_backend_detail.clear();
  iterator->second.decode_qkv_fuse_blocked = false;
  iterator->second.decode_qkv_fuse_detail.clear();
  iterator->second.decode_ffn_fuse_blocked = false;
  iterator->second.decode_ffn_fuse_detail.clear();
  iterator->second.decode_layer_tail_fuse_blocked = false;
  iterator->second.decode_layer_tail_fuse_detail.clear();
  iterator->second.prefill_projection_metrics.clear();
  iterator->second.decode_phase_metrics.clear();
  const JsonObject *kv_plan = find_object_field(params, "kvPlan");
  if (kv_plan != nullptr) {
    iterator->second.kv_cache_bytes = static_cast<std::uint64_t>(json_number_or_default(*kv_plan, "estimatedBytes", 0));
  }
  iterator->second.max_tokens = static_cast<std::uint64_t>(json_number_or_default(params, "maxTokens", 0));
  iterator->second.stop_sequences = json_array_to_strings(find_object_value(params, "stopSequences"));
  BatchExecutionConfig batch_execution;
  batch_execution.prefill_projection_metrics = &iterator->second.prefill_projection_metrics;
  bool allow_gpu_decode_from_plan = true;
  std::string gpu_decode_plan_detail;
  std::string gpu_decode_policy = "allowed";
  if (const JsonObject *profile = find_object_field(params, "profile"); profile != nullptr) {
    if (const JsonObject *native_runtime = find_object_field(*profile, "nativeRuntime"); native_runtime != nullptr) {
      gpu_decode_policy = to_lower(json_string_or_default(*native_runtime, "gpuDecodePolicy", "allowed"));
      if (gpu_decode_policy != "required"
        && gpu_decode_policy != "disabled"
        && gpu_decode_policy != "allowed") {
        throw WorkerError(
          "NATIVE_WORKER_ERROR",
          "nativeRuntime.gpuDecodePolicy must be 'required', 'allowed', or 'disabled'.");
      }
      batch_execution.allow_hybrid_gpu = json_bool_or_default(*native_runtime, "enableHybridGpuPrefill", false);
      batch_execution.gpu_tile_rows = std::max<std::uint64_t>(
        16,
        static_cast<std::uint64_t>(json_number_or_default(*native_runtime, "gpuTileRows", 512)));
      batch_execution.min_gpu_prefill_tokens = std::max<std::uint64_t>(
        1,
        static_cast<std::uint64_t>(json_number_or_default(*native_runtime, "minHybridGpuPrefillTokens", 12)));
    }
  }
  iterator->second.gpu_decode_policy = gpu_decode_policy;
  if (const JsonObject *execution_plan = find_object_field(params, "executionPlan"); execution_plan != nullptr) {
    const std::string execution_mode = json_string_or_default(*execution_plan, "mode");
    if (!batch_execution.allow_hybrid_gpu && execution_mode != "cpu-only") {
      batch_execution.allow_hybrid_gpu = true;
    }
    if (execution_mode == "cpu-only") {
      batch_execution.allow_hybrid_gpu = false;
    }
    if (const JsonObject *execution_settings = find_object_field(*execution_plan, "execution"); execution_settings != nullptr) {
      batch_execution.min_gpu_prefill_tokens = std::max<std::uint64_t>(
        1,
        static_cast<std::uint64_t>(json_number_or_default(*execution_settings, "minHybridGpuPrefillTokens", 12)));
      allow_gpu_decode_from_plan = json_bool_or_default(*execution_settings, "allowGpuDecode", true);
      gpu_decode_plan_detail = json_string_or_default(*execution_settings, "decodeReason");
    }
  }
  if (gpu_decode_policy == "disabled") {
    batch_execution.allow_hybrid_gpu = false;
    batch_execution.detail = "Direct3D 12 decode was disabled by nativeRuntime.gpuDecodePolicy.";
  }
  if (batch_execution.allow_hybrid_gpu && !d3d12_batch_matmul_capabilities().supported) {
    batch_execution.allow_hybrid_gpu = false;
    batch_execution.detail = d3d12_batch_matmul_capabilities().reason;
  }
  if (gpu_decode_policy == "required") {
    if (!batch_execution.allow_hybrid_gpu) {
      throw_required_gpu_decode_failure(
        "Session prefill",
        batch_execution.detail.empty()
          ? "the execution plan did not enable the GPU lane"
          : batch_execution.detail);
    }
    if (state.loaded_model == nullptr || !gpu_decode_model_quantization_eligible(*state.loaded_model)) {
      throw_required_gpu_decode_failure(
        "Session prefill",
        state.loaded_model == nullptr
          ? "no model is loaded"
          : gpu_decode_quantization_gate_detail(*state.loaded_model));
    }
  }
  g_gpu_decode_required = gpu_decode_policy == "required";
  // Decode follows the execution plan and device capability, not the prefill
  // size heuristic. Short prompts can stay on CPU for prefill while every
  // subsequent eligible matrix-vector decode remains on the resident GPU lane.
  g_gpu_lane_session_enabled.store(
    batch_execution.allow_hybrid_gpu && allow_gpu_decode_from_plan,
    std::memory_order_relaxed);
  if (!allow_gpu_decode_from_plan && !gpu_decode_plan_detail.empty()) {
    iterator->second.decode_backend_detail = gpu_decode_plan_detail;
  }
  if (batch_execution.allow_hybrid_gpu
    && iterator->second.prompt_token_ids.size() < batch_execution.min_gpu_prefill_tokens) {
    batch_execution.allow_hybrid_gpu = false;
    batch_execution.detail = "Direct3D 12 hybrid prefill stayed on CPU because the prompt token count is below the measured GPU-offload threshold; GPU decode remains enabled.";
  }
  std::uint64_t kv_prefix_reused_tokens = 0;
  if (state.loaded_model != nullptr && layered_llama_graph_executable(*state.loaded_model)) {
    const std::vector<std::uint64_t> prompt_token_ids_copy = iterator->second.prompt_token_ids;
    kv_prefix_reused_tokens = try_reuse_kv_prefix(
      iterator->second,
      previous_sequence_token_ids,
      previous_cached_token_count,
      prompt_token_ids_copy,
      state.loaded_model->tensor_index.layers.size());
    if (kv_prefix_reused_tokens == 0) {
      reset_layered_session_state(iterator->second, state.loaded_model->tensor_index.layers.size());
    }
    const std::vector<std::uint64_t> pending_token_ids(
      prompt_token_ids_copy.begin() + static_cast<std::ptrdiff_t>(kv_prefix_reused_tokens),
      prompt_token_ids_copy.end());
    // The batch path only wins when the hybrid GPU lane takes the batched
    // matmuls. Below the GPU-offload prompt threshold the whole batch would
    // run on CPU, which loses badly to the per-token loop's fused resident
    // GPU decode projections — keep short prompts on the per-token path.
    // Which branch runs decides TTFT, and it was not observable. Measured
    // 2026-07-30: TTFT scales linearly with prompt length at ~50 ms per prompt
    // token (765 ms at ~6 tokens, 6911 ms at ~120), which is the per-token
    // fallback doing a full forward pass per token. Two labels under
    // LIZARD_NATIVE_DECODE_PROFILE=1 say which path was taken and what it cost,
    // so the gate conditions can be judged instead of guessed.
    const auto prefill_path_started = std::chrono::steady_clock::now();
    // These two labels were print-only, so which branch ran was visible only in
    // the worker's stderr - which NativeWorkerProcessClient discards. Measuring
    // TTFT from the JS side therefore could not tell a batched prefill from a
    // per-token fallback, and `prefillBackendDetail` cannot substitute: it is
    // batch_execution.detail, a CAPABILITY string set before this branch, so it
    // reports "resident quantized batched prefill" either way. Recording a
    // metric puts the taken branch in `decodePhaseMetrics` on the response.
    const std::uint64_t prefill_gpu_use_count_before = g_decode_gpu_use_count;
    const std::string prefill_gpu_detail_before = g_decode_gpu_detail;
    if (pending_token_ids.size() > 1
      && state.loaded_model->llama_layer_shapes_ready
      && batch_execution.allow_hybrid_gpu) {
      execute_llama_prefill_batch(
        *state.loaded_model,
        iterator->second,
        pending_token_ids,
        &batch_execution);
      record_decode_phase_metric(
        iterator->second,
        "prefill-batched",
        -1,
        prefill_path_started,
        prefill_gpu_use_count_before,
        prefill_gpu_detail_before);
    } else {
      for (std::uint64_t token_id : pending_token_ids) {
        execute_llama_consume_token(*state.loaded_model, iterator->second, token_id);
      }
      record_decode_phase_metric(
        iterator->second,
        pending_token_ids.size() <= 1 ? "prefill-single-token"
        : !state.loaded_model->llama_layer_shapes_ready ? "prefill-per-token-no-shapes"
        : "prefill-per-token-gpu-declined",
        -1,
        prefill_path_started,
        prefill_gpu_use_count_before,
        prefill_gpu_detail_before);
    }
    iterator->second.prefill_complete = !prompt_token_ids_copy.empty();
  } else if (state.loaded_model != nullptr && layered_gemma_graph_executable(*state.loaded_model)) {
    const std::vector<std::uint64_t> prompt_token_ids_copy = iterator->second.prompt_token_ids;
    kv_prefix_reused_tokens = try_reuse_kv_prefix(
      iterator->second,
      previous_sequence_token_ids,
      previous_cached_token_count,
      prompt_token_ids_copy,
      state.loaded_model->tensor_index.layers.size());
    if (kv_prefix_reused_tokens == 0) {
      reset_layered_session_state(iterator->second, state.loaded_model->tensor_index.layers.size());
    }
    const std::vector<std::uint64_t> pending_token_ids(
      prompt_token_ids_copy.begin() + static_cast<std::ptrdiff_t>(kv_prefix_reused_tokens),
      prompt_token_ids_copy.end());
    if (pending_token_ids.size() > 1) {
      execute_gemma_prefill_batch(
        *state.loaded_model,
        iterator->second,
        pending_token_ids,
        &batch_execution);
    } else if (!pending_token_ids.empty()) {
      execute_gemma_consume_token(
        *state.loaded_model,
        iterator->second,
        pending_token_ids.front());
    }
    iterator->second.prefill_complete = !prompt_token_ids_copy.empty();
  } else if (state.loaded_model != nullptr && layered_family_graph_executable(*state.loaded_model)) {
    const std::vector<std::uint64_t> prompt_token_ids_copy = iterator->second.prompt_token_ids;
    kv_prefix_reused_tokens = try_reuse_kv_prefix(
      iterator->second,
      previous_sequence_token_ids,
      previous_cached_token_count,
      prompt_token_ids_copy,
      state.loaded_model->tensor_index.layers.size());
    if (kv_prefix_reused_tokens == 0) {
      reset_layered_session_state(iterator->second, state.loaded_model->tensor_index.layers.size());
    }
    const std::vector<std::uint64_t> pending_token_ids(
      prompt_token_ids_copy.begin() + static_cast<std::ptrdiff_t>(kv_prefix_reused_tokens),
      prompt_token_ids_copy.end());
    for (std::uint64_t token_id : pending_token_ids) {
      execute_family_consume_token(*state.loaded_model, iterator->second, token_id);
    }
    iterator->second.prefill_complete = !prompt_token_ids_copy.empty();
  } else {
    iterator->second.cached_token_count = static_cast<std::uint64_t>(iterator->second.sequence_token_ids.size());
  }
  if (batch_execution.hybrid_gpu_used) {
    iterator->second.native_backend = batch_execution.native_backend;
    iterator->second.native_backend_detail = batch_execution.detail;
    iterator->second.prefill_backend = batch_execution.native_backend;
    iterator->second.prefill_backend_detail = batch_execution.detail;
  } else if (!batch_execution.detail.empty()) {
    iterator->second.native_backend_detail = batch_execution.detail;
    iterator->second.prefill_backend_detail = batch_execution.detail;
  }
  JsonArray prompt_token_ids;
  for (std::uint64_t token_id : iterator->second.prompt_token_ids) {
    prompt_token_ids.push_back(JsonValue(token_id));
  }
  const auto prefill_finished_at = std::chrono::steady_clock::now();
  const std::uint64_t prefill_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
    1,
    std::chrono::duration_cast<std::chrono::milliseconds>(prefill_finished_at - prefill_started_at).count()));
  iterator->second.last_prefill_ms = prefill_ms;
  const std::uint64_t cached_token_count = state.loaded_model != nullptr && native_graph_executable(*state.loaded_model)
    ? iterator->second.cached_token_count
    : static_cast<std::uint64_t>(iterator->second.sequence_token_ids.size());
  JsonObject result = {
    {"sessionId", JsonValue(session_id)},
    {"nativeBackend", JsonValue(iterator->second.native_backend)},
    {"prefillMs", JsonValue(prefill_ms)},
    {"kvCacheBytes", JsonValue(iterator->second.kv_cache_bytes)},
    {"promptTokenIds", JsonValue(prompt_token_ids)},
    {"promptTokenCount", JsonValue(static_cast<std::uint64_t>(iterator->second.prompt_token_ids.size()))},
    {"cachedTokenCount", JsonValue(cached_token_count)},
    {"promptTokenMode", JsonValue(iterator->second.prompt_token_mode.empty() ? "unavailable" : iterator->second.prompt_token_mode)},
    {"kvPrefixReusedTokens", JsonValue(kv_prefix_reused_tokens)},
    {"sessionMetrics", JsonValue(build_session_metrics_object(iterator->second))},
    // The generate response already carries these, but the prefill branch is
    // decided HERE and generate clears the vector before it runs, so the phase
    // that names the taken branch is only observable on this response.
    {"decodePhaseMetrics", JsonValue(build_decode_phase_metrics_array(iterator->second))},
  };
  if (state.loaded_model != nullptr && !iterator->second.prompt_token_ids.empty()) {
    std::vector<float> preview_values;
    std::string native_model_status;
    if (zero_layer_graph_executable(*state.loaded_model)) {
      const TensorInfo &token_embeddings = tensor_at(*state.loaded_model, state.loaded_model->tensor_index.token_embeddings);
      preview_values = tensor_row_f32(
        *state.loaded_model,
        token_embeddings,
        iterator->second.prompt_token_ids.back());
      native_model_status = "minimal-graph-prefilled";
    } else if (layered_llama_graph_executable(*state.loaded_model)) {
      preview_values = iterator->second.last_hidden_state;
      native_model_status = "llama-graph-prefilled";
    } else if (layered_gemma_graph_executable(*state.loaded_model)) {
      preview_values = iterator->second.last_hidden_state;
      native_model_status = "gemma-graph-prefilled";
    } else if (layered_family_graph_executable(*state.loaded_model)) {
      const auto family_spec = resolve_family_spec(state.loaded_model->model);
      preview_values = iterator->second.last_hidden_state;
      native_model_status = (family_spec.has_value() ? family_spec->family : "family") + "-graph-prefilled";
    }
    if (!preview_values.empty()) {
      JsonArray preview;
      const std::size_t preview_count = std::min<std::size_t>(8, preview_values.size());
      for (std::size_t index = 0; index < preview_count; index += 1) {
        preview.push_back(JsonValue(static_cast<double>(preview_values[index])));
      }
      result.emplace("embeddingPreview", JsonValue(preview));
      result.emplace("nativeModelStatus", JsonValue(native_model_status));
    }
  }
  return result;
}

static JsonObject handle_generate_or_decode(WorkerState &state, const JsonObject &params) {
  if (state.loaded_model == nullptr) {
    throw WorkerError("NATIVE_WORKER_ERROR", "No native model is loaded.");
  }
  const std::string session_id = json_string_or_default(params, "sessionId");
  auto iterator = state.sessions.find(session_id);
  if (iterator == state.sessions.end()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Unknown native worker session.");
  }
  if (iterator != state.sessions.end() && iterator->second.cancelled) {
    throw WorkerError("NATIVE_CANCELLED", "The native session was cancelled.");
  }
  if (!iterator->second.sampling.initialized
    || iterator->second.sequence_token_ids.size() <= iterator->second.prompt_token_ids.size()) {
    iterator->second.sampling = parse_sampling_state(params);
  }
  iterator->second.stream_tokens = json_bool_or_default(params, "emitTokenEvents", false);
  reset_decode_resource_metrics(iterator->second);
  const ProcessResourceSnapshot decode_resource_before = read_process_resource_snapshot();
  const auto finalize_result = [&](JsonObject result) {
    const ProcessResourceSnapshot decode_resource_after = read_process_resource_snapshot();
    append_decode_resource_metrics(
      iterator->second,
      decode_resource_before,
      decode_resource_after,
      iterator->second.last_decode_ms,
      iterator->second.last_decode_token_count,
      false);
    result.insert_or_assign(
      "sessionMetrics",
      JsonValue(build_session_metrics_object(iterator->second)));
    result.insert_or_assign(
      "nativeBackend",
      JsonValue(iterator->second.decode_backend.empty() ? "cpu" : iterator->second.decode_backend));
    if (!iterator->second.decode_backend_detail.empty()) {
      result.insert_or_assign("nativeBackendDetail", JsonValue(iterator->second.decode_backend_detail));
    }
    result.insert_or_assign(
      "prefillBackend",
      JsonValue(iterator->second.prefill_backend.empty() ? "cpu" : iterator->second.prefill_backend));
    if (!iterator->second.prefill_backend_detail.empty()) {
      result.insert_or_assign("prefillBackendDetail", JsonValue(iterator->second.prefill_backend_detail));
    }
    result.insert_or_assign(
      "decodeBackend",
      JsonValue(iterator->second.decode_backend.empty() ? "cpu" : iterator->second.decode_backend));
    if (!iterator->second.decode_backend_detail.empty()) {
      result.insert_or_assign("decodeBackendDetail", JsonValue(iterator->second.decode_backend_detail));
    }
    state.last_metrics = JsonValue(result);
    return result;
  };
  if (zero_layer_graph_executable(*state.loaded_model)) {
    const std::uint64_t max_tokens = static_cast<std::uint64_t>(json_number_or_default(params, "maxTokens", 0));
    const std::uint64_t prefill_ms = static_cast<std::uint64_t>(json_number_or_default(params, "prefillMs", 1));
    return finalize_result(execute_zero_layer_generate(
      *state.loaded_model,
      iterator->second,
      max_tokens == 0 ? 1 : max_tokens,
      prefill_ms == 0 ? 1 : prefill_ms));
  }
  if (layered_llama_graph_executable(*state.loaded_model)) {
    const std::uint64_t max_tokens = static_cast<std::uint64_t>(json_number_or_default(params, "maxTokens", 0));
    const std::uint64_t prefill_ms = static_cast<std::uint64_t>(json_number_or_default(params, "prefillMs", 1));
    return finalize_result(execute_llama_layered_generate(
      *state.loaded_model,
      iterator->second,
      max_tokens == 0 ? 1 : max_tokens,
      prefill_ms == 0 ? 1 : prefill_ms));
  }
  if (layered_gemma_graph_executable(*state.loaded_model)) {
    const std::uint64_t max_tokens = static_cast<std::uint64_t>(json_number_or_default(params, "maxTokens", 0));
    const std::uint64_t prefill_ms = static_cast<std::uint64_t>(json_number_or_default(params, "prefillMs", 1));
    const bool cache_final_token = json_bool_or_default(params, "cacheFinalToken", false);
    return finalize_result(execute_gemma_layered_generate(
      *state.loaded_model,
      iterator->second,
      max_tokens == 0 ? 1 : max_tokens,
      prefill_ms == 0 ? 1 : prefill_ms,
      cache_final_token));
  }
  if (layered_family_graph_executable(*state.loaded_model)) {
    const std::uint64_t max_tokens = static_cast<std::uint64_t>(json_number_or_default(params, "maxTokens", 0));
    const std::uint64_t prefill_ms = static_cast<std::uint64_t>(json_number_or_default(params, "prefillMs", 1));
    return finalize_result(execute_family_layered_generate(
      *state.loaded_model,
      iterator->second,
      max_tokens == 0 ? 1 : max_tokens,
      prefill_ms == 0 ? 1 : prefill_ms));
  }
  const JsonObject *executor = find_object_field(state.loaded_model->compatibility, "executor");
  const std::string executor_status = executor == nullptr
    ? "unsupported"
    : json_string_or_default(*executor, "status", "unsupported");
  const std::string executor_reason = executor == nullptr
    ? "The loaded GGUF is outside the native worker's current execution subset."
    : json_string_or_default(*executor, "reason", "The loaded GGUF is outside the native worker's current execution subset.");
  throw WorkerError(
    executor_status == "planned" ? "NATIVE_EXECUTOR_UNIMPLEMENTED" : "NATIVE_MODEL_UNSUPPORTED",
    executor_reason,
    JsonObject {
      {"model", JsonValue(build_model_object(state.loaded_model->model))},
      {"compatibility", JsonValue(state.loaded_model->compatibility)},
      {"executorState", JsonValue(build_tensor_index_summary(*state.loaded_model))},
    });
}

// EnhancementRun_04 Phase 4 -- one decode step across many sessions.
//
// Advances every named session by exactly one token in a single pass, so each
// layer's weights are read once for the whole batch instead of once per
// sequence. Greedy only: this is the throughput measurement path, and a
// sampler would put a second variable into a number whose whole purpose is to
// isolate the effect of batching.
//
// Sessions must already be prefilled. A session with no logits has nothing to
// continue from, and silently skipping it would make the batch size reported
// in telemetry a lie.
static JsonObject handle_decode_batch(WorkerState &state, const JsonObject &params) {
  if (state.loaded_model == nullptr) {
    throw WorkerError("NATIVE_WORKER_ERROR", "decode_batch requires a loaded model.");
  }
  const std::vector<std::string> session_ids =
    json_array_to_strings(find_object_value(params, "sessionIds"));
  if (session_ids.empty()) {
    throw WorkerError("NATIVE_WORKER_ERROR", "decode_batch requires a non-empty sessionIds array.");
  }

  std::vector<SessionState *> sessions;
  std::vector<std::uint64_t> tokens;
  sessions.reserve(session_ids.size());
  tokens.reserve(session_ids.size());
  for (const std::string &session_id : session_ids) {
    const auto iterator = state.sessions.find(session_id);
    if (iterator == state.sessions.end()) {
      throw WorkerError("NATIVE_WORKER_ERROR", "decode_batch received an unknown session: " + session_id);
    }
    if (iterator->second.last_logits.empty()) {
      throw WorkerError(
        "NATIVE_WORKER_ERROR",
        "decode_batch requires every session to be prefilled; " + session_id + " has no logits.");
    }
    sessions.push_back(&iterator->second);
    tokens.push_back(static_cast<std::uint64_t>(argmax_index(iterator->second.last_logits)));
  }

  // The GPU lane has to be handed in explicitly. Running this path with a null
  // execution config pins every projection to the CPU: the first measurement of
  // batched decode read 0.48 tok/s against the normal path's 6.50 for exactly
  // that reason. `min_gpu_prefill_tokens` is set to 1 because it is a PREFILL
  // heuristic -- a decode step is one row per sequence by definition and would
  // never clear a prefill-sized threshold.
  BatchExecutionConfig execution_config;
  execution_config.allow_hybrid_gpu = gpu_decode_lane_enabled_for_model(*state.loaded_model)
    && d3d12_batch_matmul_capabilities().supported;
  execution_config.min_gpu_prefill_tokens = 1;

  // Exact deterministic request coalescing. The server already restores
  // repeated prompts from one isolated prefill snapshot; those clones carry
  // the same equivalence generation. While their full token histories and
  // current logits remain byte-identical, one physical greedy step is enough
  // for the whole group. Different prompts, independently-prefilled sessions,
  // or sessions at different positions remain separate rows.
  std::vector<std::vector<std::size_t>> equivalent_groups;
  for (std::size_t row = 0; row < sessions.size(); row += 1) {
    bool joined = false;
    for (std::vector<std::size_t> &group : equivalent_groups) {
      const std::size_t representative = group.front();
      const SessionState &left = *sessions[representative];
      const SessionState &right = *sessions[row];
      if (left.decode_equivalence_id != 0
        && left.decode_equivalence_id == right.decode_equivalence_id
        && left.cached_token_count == right.cached_token_count
        && tokens[representative] == tokens[row]
        && left.sequence_token_ids == right.sequence_token_ids
        && left.last_logits == right.last_logits) {
        group.push_back(row);
        joined = true;
        break;
      }
    }
    if (!joined) {
      equivalent_groups.push_back({row});
    }
  }
  std::vector<SessionState *> effective_sessions;
  std::vector<std::uint64_t> effective_tokens;
  effective_sessions.reserve(equivalent_groups.size());
  effective_tokens.reserve(equivalent_groups.size());
  for (const std::vector<std::size_t> &group : equivalent_groups) {
    effective_sessions.push_back(sessions[group.front()]);
    effective_tokens.push_back(tokens[group.front()]);
  }

  const ProcessResourceSnapshot decode_resource_before = read_process_resource_snapshot();
  const auto started_at = std::chrono::steady_clock::now();
  FloatBatch row_logits;
  if (effective_sessions.size() == 1) {
    // A one-row batch is the one case batching cannot help, and taking the
    // batched path for it is actively worse: the GPU resident lane requires
    // more than one row (`batch_matmul_gpu_resident_eligible`), so a single row
    // falls off it entirely and lands on the CPU at ~2 s/step against the
    // matvec kernel's ~150 ms. Route it to the path built for it.
    execute_llama_consume_token(
      *state.loaded_model, *effective_sessions[0], effective_tokens[0]);
  } else {
    execute_llama_decode_batch(
      *state.loaded_model,
      effective_sessions,
      effective_tokens,
      &execution_config,
      &row_logits);
  }

  // Followers retain request-local limits, stop state, cancellation and
  // accounting, but receive the representative's new model/KV state. This
  // keeps later step-boundary departures safe: any remaining follower is a
  // complete standalone session, not an alias to a retired leader.
  for (const std::vector<std::size_t> &group : equivalent_groups) {
    if (group.size() < 2) continue;
    const SessionState &source = *sessions[group.front()];
    for (std::size_t member = 1; member < group.size(); member += 1) {
      SessionState &target = *sessions[group[member]];
      target.sequence_token_ids = source.sequence_token_ids;
      target.layered_kv_cache = source.layered_kv_cache;
      target.last_hidden_state = source.last_hidden_state;
      target.last_logits = source.last_logits;
      target.last_gpu_argmax_token = source.last_gpu_argmax_token;
      target.cached_token_count = source.cached_token_count;
      target.prefill_complete = source.prefill_complete;
      target.native_backend = source.native_backend;
      target.native_backend_detail = source.native_backend_detail;
      target.decode_backend = source.decode_backend;
      target.decode_backend_detail = source.decode_backend_detail;
      target.decode_qkv_fuse_blocked = source.decode_qkv_fuse_blocked;
      target.decode_qkv_fuse_detail = source.decode_qkv_fuse_detail;
      target.decode_ffn_fuse_blocked = source.decode_ffn_fuse_blocked;
      target.decode_ffn_fuse_detail = source.decode_ffn_fuse_detail;
      target.decode_layer_tail_fuse_blocked = source.decode_layer_tail_fuse_blocked;
      target.decode_layer_tail_fuse_detail = source.decode_layer_tail_fuse_detail;
      target.decode_phase_metrics = source.decode_phase_metrics;
    }
  }
  const std::uint64_t elapsed_ms = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started_at).count());
  const ProcessResourceSnapshot decode_resource_after = read_process_resource_snapshot();

  // Termination lives here, not in the caller. EOS ids, the "<turn|>" marker
  // and the raw-BPE whitespace decoding the stop-tail scan needs are all
  // runner-side facts; re-deriving them in JS would mean two implementations of
  // "when is an answer finished" that drift apart, and the failure mode is a
  // model that keeps writing past its own turn-ender.
  const auto eos_token_id = json_to_u64(find_object_value(state.loaded_model->model.special_tokens, "eos"));
  // Same bound the single-sequence loop uses: only the last few dozen
  // characters can contain a stop marker, and letting the tail grow unbounded
  // would make every step scan the whole answer.
  constexpr std::size_t kBatchStopScanTailMax = 64;

  JsonArray emitted;
  emitted.reserve(session_ids.size());
  std::uint64_t cohort_decoded_tokens = 0;
  for (std::size_t row = 0; row < session_ids.size(); row += 1) {
    const auto eos_token_id = json_to_u64(find_object_value(state.loaded_model->model.special_tokens, "eos"));
    const std::uint64_t token_id = tokens[row];
    const std::string token_text = token_id < state.loaded_model->model.token_vocabulary.size()
      ? state.loaded_model->model.token_vocabulary[static_cast<std::size_t>(token_id)]
      : std::string();
    if ((!eos_token_id.has_value() || token_id != eos_token_id.value()) && token_text != "<turn|>") {
      cohort_decoded_tokens += 1;
    }
  }
  for (std::size_t row = 0; row < session_ids.size(); row += 1) {
    SessionState &session = *sessions[row];
    session.generated_token_count += 1;
    session.last_decode_token_count = 1;
    session.last_decode_ms = elapsed_ms;

    const std::uint64_t token_id = tokens[row];
    const std::string token_text = token_id < state.loaded_model->model.token_vocabulary.size()
      ? state.loaded_model->model.token_vocabulary[static_cast<std::size_t>(token_id)]
      : std::string();

    std::string finish_reason;
    // EOS and the native turn marker are model control tokens and always end a
    // completion. Extra textual stop sequences remain request-controlled.
    // This is deliberately identical to the single-sequence path: a real HTTP
    // benchmark must not spend time decoding hidden post-EOS tokens.
    const bool apply_text_stop = !session.stop_sequences.empty();
    // EOS is consumed, not shown -- the single-sequence loop breaks BEFORE
    // recording it, so reporting it would make a batched answer one token
    // longer than the same prompt decoded alone. A stop SEQUENCE is different
    // and IS recorded: it is part of the visible answer and the JS filter trims
    // the marker itself.
    bool suppressed = false;
    if ((eos_token_id.has_value() && token_id == eos_token_id.value()) || token_text == "<turn|>") {
      finish_reason = "eos";
      suppressed = true;
    } else if (apply_text_stop) {
      // Only the tail can contain a marker, and a marker split across tokens
      // must still be caught -- so the scan is over accumulated decoded text,
      // bounded, exactly as the single-sequence loop does it.
      session.batch_stop_scan_tail += decode_bpe_whitespace(token_text);
      if (session.batch_stop_scan_tail.size() > kBatchStopScanTailMax) {
        session.batch_stop_scan_tail.erase(
          0, session.batch_stop_scan_tail.size() - kBatchStopScanTailMax);
      }
      for (const std::string &stop_marker : session.stop_sequences) {
        if (!stop_marker.empty()
          && session.batch_stop_scan_tail.find(stop_marker) != std::string::npos) {
          finish_reason = "stop-sequence";
          break;
        }
      }
    }
    if (finish_reason.empty()
      && session.max_tokens > 0
      && session.generated_token_count >= session.max_tokens) {
      finish_reason = "max-tokens";
    }
    if (session.cancelled) {
      finish_reason = "cancelled";
    }

    if (suppressed) {
      // Not part of the answer, so it must not count toward the budget either.
      session.generated_token_count -= 1;
    }

    append_decode_resource_metrics(
      session,
      decode_resource_before,
      decode_resource_after,
      elapsed_ms,
      cohort_decoded_tokens,
      true);

    emitted.push_back(JsonValue(JsonObject {
      {"sessionId", JsonValue(session_ids[row])},
      {"tokenId", JsonValue(token_id)},
      {"token", JsonValue(token_text)},
      {"suppressed", JsonValue(suppressed)},
      {"cachedTokenCount", JsonValue(session.cached_token_count)},
      {"generatedTokenCount", JsonValue(session.generated_token_count)},
      {"finished", JsonValue(!finish_reason.empty())},
      {"finishReason", finish_reason.empty() ? JsonValue() : JsonValue(finish_reason)},
    }));
  }

  return {
    {"batchSize", JsonValue(static_cast<std::uint64_t>(session_ids.size()))},
    {"effectiveRows", JsonValue(static_cast<std::uint64_t>(effective_sessions.size()))},
    {"coalescedSessions", JsonValue(
      static_cast<std::uint64_t>(session_ids.size() - effective_sessions.size()))},
    {"stepMs", JsonValue(elapsed_ms)},
    {"backend", JsonValue(
      execution_config.hybrid_gpu_used
        ? "directx-hybrid"
        : (effective_sessions.empty() || effective_sessions.front()->decode_backend.empty()
          ? "cpu"
          : effective_sessions.front()->decode_backend))},
    {"backendDetail", JsonValue(
      !execution_config.detail.empty()
        ? execution_config.detail
        : (effective_sessions.empty()
          ? std::string()
          : effective_sessions.front()->decode_backend_detail))},
    // The premise assertion. Weight traffic per STEP must be flat in B; if it
    // grows with the batch then batching has bought nothing and the phase has
    // failed on its own terms, whatever the tok/s says.
    {"weightBytesPerStep", JsonValue(decode_bandwidth_model().weight_bytes_per_token)},
    {"tokens", JsonValue(emitted)},
  };
}

static JsonObject handle_cancel(WorkerState &state, const JsonObject &params) {
  const std::string session_id = json_string_or_default(params, "sessionId");
  auto iterator = state.sessions.find(session_id);
  if (iterator == state.sessions.end()) {
    return {
      {"cancelled", JsonValue(false)},
    };
  }
  iterator->second.cancelled = true;
  return {
    {"cancelled", JsonValue(true)},
    {"sessionMetrics", JsonValue(build_session_metrics_object(iterator->second))},
  };
}

static JsonObject handle_metrics(const WorkerState &state) {
  JsonArray sessions;
  std::uint64_t total_native_session_bytes = 0;
  std::uint64_t total_kv_cache_bytes = 0;
  for (const auto &[session_id, session] : state.sessions) {
    sessions.push_back(JsonValue(build_session_metrics_object(session)));
    total_native_session_bytes += estimate_session_native_bytes(session);
    total_kv_cache_bytes += session.kv_cache_bytes;
  }
  JsonObject metrics = {
    {"runtimeProvider", JsonValue("lizard-native")},
    {"nativeBackend", JsonValue(ggml_cpu_backend_effective_name())},
    {"cpuBackend", build_cpu_backend_object()},
    {"nativeModelStatus", JsonValue(native_model_status(state.loaded_model.get()))},
    {"executionReady", JsonValue(state.loaded_model != nullptr && native_graph_executable(*state.loaded_model))},
    {"activeSessionCount", JsonValue(static_cast<std::uint64_t>(state.sessions.size()))},
    {"totalNativeSessionBytes", JsonValue(total_native_session_bytes)},
    {"totalKvCacheBytes", JsonValue(total_kv_cache_bytes)},
    {"sessions", JsonValue(sessions)},
  };
  // EnhancementRun_04 Phase 2. Always present, so a reader can tell "the step
  // loop is not enabled" from "it is enabled and idle" -- an absent field would
  // conflate the two.
  const EngineLoopTelemetry engine = engine_loop_telemetry();
  metrics.emplace("engineLoop", JsonValue(JsonObject {
    {"running", JsonValue(engine.running)},
    {"steps", JsonValue(engine.steps)},
    {"commandsHandled", JsonValue(engine.commands_handled)},
    {"queueDepthMax", JsonValue(engine.queue_depth_max)},
    {"readerBlockedCount", JsonValue(engine.reader_blocked_count)},
    {"queueCapacity", JsonValue(static_cast<std::uint64_t>(engine_loop_queue_capacity()))},
  }));
  if (state.loaded_model != nullptr) {
    metrics.emplace("model", JsonValue(build_model_object(state.loaded_model->model)));
    metrics.emplace("executorState", JsonValue(build_tensor_index_summary(*state.loaded_model)));
  }
  // The aggregate session view must always be present; the most recent
  // generate/decode result stays available for callers that want it.
  if (!state.last_metrics.is_null()) {
    metrics.emplace("lastRun", state.last_metrics);
  }
  return metrics;
}

static JsonObject handle_tensor_benchmark(const WorkerState &state, const JsonObject &params) {
  if (state.loaded_model == nullptr) {
    throw WorkerError("NATIVE_WORKER_ERROR", "No native model is loaded.");
  }
  const std::string requested_tensor_name = json_string_or_default(params, "tensorName");
  const std::string requested_backend = to_lower(json_string_or_default(params, "backend", "cpu"));
  if (requested_backend != "cpu" && requested_backend != "gpu" && requested_backend != "compare") {
    throw WorkerError("NATIVE_WORKER_ERROR", "tensor_benchmark backend must be cpu, gpu, or compare.");
  }
  std::size_t tensor_index = k_missing_tensor_index;
  if (!requested_tensor_name.empty()) {
    for (std::size_t index = 0; index < state.loaded_model->model.tensors.size(); index += 1) {
      if (state.loaded_model->model.tensors[index].name == requested_tensor_name) {
        tensor_index = index;
        break;
      }
    }
  } else if (requested_backend == "compare") {
    // Pick one representative projection, bounded so first-load calibration
    // cannot turn into a full-model benchmark. Prefer the largest supported
    // tensor up to 16M elements for a useful CPU/GPU signal.
    std::uint64_t best_elements = 0;
    for (std::size_t index = 0; index < state.loaded_model->model.tensors.size(); index += 1) {
      const TensorInfo &candidate = state.loaded_model->model.tensors[index];
      if (candidate.dimensions.size() != 2
        || candidate.element_count < 1048576
        || candidate.element_count > 16777216) {
        continue;
      }
      D3d12WeightFormat format = D3d12WeightFormat::F32;
      if (!d3d12_weight_format_supported(candidate.ggml_type, candidate.dimensions[0], &format)) {
        continue;
      }
      if (candidate.element_count > best_elements) {
        best_elements = candidate.element_count;
        tensor_index = index;
      }
    }
  }
  if (tensor_index == k_missing_tensor_index) {
    throw WorkerError("NATIVE_WORKER_ERROR", "No suitable tensor was found for the requested benchmark.", JsonObject {
      {"tensorName", JsonValue(requested_tensor_name)},
    });
  }

  const TensorInfo &tensor = tensor_at(*state.loaded_model, tensor_index);
  const MatrixShape shape = tensor_matrix_shape(tensor, "Tensor benchmark");
  const std::uint64_t iterations = std::min<std::uint64_t>(
    3,
    std::max<std::uint64_t>(
      1,
      static_cast<std::uint64_t>(json_number_or_default(params, "iterations", 1))));
  std::vector<float> input(static_cast<std::size_t>(shape.row_width), 0.0f);
  for (std::size_t index = 0; index < input.size(); index += 1) {
    input[index] = static_cast<float>((static_cast<int>(index % 17) - 8) / 16.0);
  }

  const auto run_cpu = [&]() {
    std::vector<float> output;
    double checksum = 0.0;
    const bool previous_gpu_state = g_gpu_lane_session_enabled.exchange(false, std::memory_order_relaxed);
    const auto started_at = std::chrono::steady_clock::now();
    for (std::uint64_t iteration = 0; iteration < iterations; iteration += 1) {
      output = matrix_vector_mul_tensor(*state.loaded_model, tensor, shape, input, "Tensor calibration CPU");
      for (float value : output) {
        checksum += static_cast<double>(value);
      }
    }
    const auto finished_at = std::chrono::steady_clock::now();
    g_gpu_lane_session_enabled.store(previous_gpu_state, std::memory_order_relaxed);
    const std::uint64_t elapsed_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
      1,
      std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count()));
    return std::make_tuple(std::move(output), checksum, elapsed_ms);
  };

  const auto [cpu_output, cpu_checksum, cpu_elapsed_ms] = run_cpu();
  std::vector<float> output = cpu_output;
  double checksum = cpu_checksum;
  std::uint64_t elapsed_ms = cpu_elapsed_ms;
  bool gpu_available = false;
  bool parity_passed = false;
  std::uint64_t gpu_elapsed_ms = 0;
  std::string gpu_failure;

  if (requested_backend == "gpu" || requested_backend == "compare") {
    D3d12WeightFormat format = D3d12WeightFormat::F32;
    if (shape.row_count > std::numeric_limits<std::uint32_t>::max()
      || shape.row_width > std::numeric_limits<std::uint32_t>::max()
      || !d3d12_weight_format_supported(tensor.ggml_type, shape.row_width, &format)) {
      gpu_failure = "Tensor format or dimensions are not supported by the Direct3D calibration lane.";
    } else {
      const std::uint8_t *weight_data = tensor_data_ptr(*state.loaded_model, tensor);
      D3d12ResidentWeightRef op;
      op.weight_key = weight_data;
      op.format = format;
      op.weight_data = weight_data;
      op.weight_bytes = static_cast<std::size_t>(tensor.byte_length);
      op.row_count = static_cast<std::uint32_t>(shape.row_count);
      op.row_width = static_cast<std::uint32_t>(shape.row_width);
      std::vector<float> gpu_output;
      // Untimed warm-up includes first staging and pipeline readiness. The
      // timed loop measures the reusable resident path that decode needs.
      if (d3d12_batch_matmul_resident(
        &op, 1, input, 1, next_gpu_input_identity(), &gpu_output, &gpu_failure)) {
        const auto gpu_started_at = std::chrono::steady_clock::now();
        for (std::uint64_t iteration = 0; iteration < iterations; iteration += 1) {
          if (!d3d12_batch_matmul_resident(
            &op, 1, input, 1, next_gpu_input_identity(), &gpu_output, &gpu_failure)) {
            break;
          }
        }
        const auto gpu_finished_at = std::chrono::steady_clock::now();
        if (gpu_failure.empty()) {
          gpu_available = true;
          gpu_elapsed_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(
            1,
            std::chrono::duration_cast<std::chrono::milliseconds>(gpu_finished_at - gpu_started_at).count()));
          double max_abs_difference = 0.0;
          double max_cpu_abs = 0.0;
          for (std::size_t index = 0; index < cpu_output.size() && index < gpu_output.size(); index += 1) {
            max_abs_difference = std::max(
              max_abs_difference,
              std::abs(static_cast<double>(cpu_output[index]) - static_cast<double>(gpu_output[index])));
            max_cpu_abs = std::max(max_cpu_abs, std::abs(static_cast<double>(cpu_output[index])));
          }
          parity_passed = cpu_output.size() == gpu_output.size()
            && max_abs_difference <= std::max(0.05, max_cpu_abs * 0.02);
          if (requested_backend == "gpu") {
            output = std::move(gpu_output);
            elapsed_ms = gpu_elapsed_ms;
            checksum = 0.0;
            for (float value : output) checksum += static_cast<double>(value);
          }
        }
      }
    }
  }

  JsonArray dimensions;
  for (std::uint64_t dimension : tensor.dimensions) {
    dimensions.push_back(JsonValue(dimension));
  }
  JsonArray preview;
  const std::size_t preview_count = std::min<std::size_t>(8, output.size());
  for (std::size_t index = 0; index < preview_count; index += 1) {
    preview.push_back(JsonValue(static_cast<double>(output[index])));
  }
  JsonObject result = {
    {"runtimeProvider", JsonValue("lizard-native")},
    {"nativeBackend", JsonValue(ggml_cpu_backend_effective_name())},
    {"modelArchitecture", JsonValue(state.loaded_model->model.architecture.empty() ? "unknown" : state.loaded_model->model.architecture)},
    {"modelQuantization", JsonValue(state.loaded_model->model.quantization.empty() ? "unknown" : state.loaded_model->model.quantization)},
    {"tensorName", JsonValue(tensor.name)},
    {"ggmlTypeName", JsonValue(tensor.ggml_type_name.empty() ? "unknown" : tensor.ggml_type_name)},
    {"dimensions", JsonValue(dimensions)},
    {"rowWidth", JsonValue(shape.row_width)},
    {"rowCount", JsonValue(shape.row_count)},
    {"iterations", JsonValue(iterations)},
    {"elapsedMs", JsonValue(elapsed_ms)},
    {"msPerIteration", JsonValue(static_cast<double>(elapsed_ms) / static_cast<double>(iterations))},
    {"checksum", JsonValue(checksum)},
    {"outputPreview", JsonValue(preview)},
  };
  if (requested_backend == "compare") {
    const double cpu_ms = static_cast<double>(cpu_elapsed_ms) / static_cast<double>(iterations);
    const double gpu_ms = gpu_available
      ? static_cast<double>(gpu_elapsed_ms) / static_cast<double>(iterations)
      : 0.0;
    const double speedup = gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0;
    const bool gpu_recommended = gpu_available && parity_passed && speedup >= 1.1;
    result.insert_or_assign("calibration", JsonValue(JsonObject {
      {"cpuMsPerIteration", JsonValue(cpu_ms)},
      {"gpuMsPerIteration", gpu_available ? JsonValue(gpu_ms) : JsonValue()},
      {"gpuSpeedup", gpu_available ? JsonValue(speedup) : JsonValue()},
      {"parityPassed", JsonValue(parity_passed)},
      {"recommendedBackend", JsonValue(gpu_recommended ? "gpu" : "cpu")},
      {"reason", JsonValue(gpu_recommended
        ? "GPU resident matvec passed parity and exceeded the 1.10x speedup threshold."
        : (!gpu_failure.empty()
          ? gpu_failure
          : (!parity_passed
            ? "GPU resident matvec failed the CPU parity threshold."
            : "GPU resident matvec did not exceed the 1.10x CPU speedup threshold.")))},
    }));
  }
  return result;
}

static JsonObject handle_shutdown(WorkerState &state) {
  state.shutdown_requested = true;
  return {
    {"shuttingDown", JsonValue(true)},
  };
}

// Phase 1 acceptance check from .agent/vision/01_native_vision_plan.md.
//
// Runs one prompt twice: once the ordinary way, and once with EVERY position's
// hidden state supplied by the caller - each row gathered from the very
// embedding table the normal path would have read. The two runs must produce
// identical logits, because the inputs are identical; only the route differs.
//
// That equality is the whole point. It proves the splice carries a row through
// RoPE, the KV cache and the layer stack without disturbing anything, BEFORE a
// vision tower exists to produce interesting rows. If this ever drifts, no
// image result from this runner can be trusted, and the failure would otherwise
// show up as a model that is merely slightly wrong about pictures.
static JsonObject handle_embedding_prompt_selftest(WorkerState &state, const JsonObject &params) {
  if (state.loaded_model == nullptr) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Load a model before running the embedding prompt self-test.");
  }
  const LoadedModelState &model_state = *state.loaded_model;
  std::vector<std::uint64_t> token_ids = json_array_to_u64s(find_object_value(params, "tokenIds"));
  if (token_ids.empty()) {
    // A short, deterministic sequence is enough: this measures plumbing, not
    // language. Token 1 is BOS in every vocabulary we load.
    token_ids = {1, 2, 3, 4, 5, 6, 7, 8};
  }

  // Gemma and llama have separate execution paths, and the splice has to hold
  // in whichever one this model actually uses.
  const bool gemma_path = is_gemma_architecture(model_state.model.architecture);

  SessionState baseline_session;
  FloatBatch baseline_logits;
  if (gemma_path) {
    execute_gemma_prefill_batch(model_state, baseline_session, token_ids, nullptr, &baseline_logits);
  } else {
    execute_llama_prefill_batch(model_state, baseline_session, token_ids, nullptr, &baseline_logits);
  }

  // The same hidden states the baseline just built, handed in from outside.
  // On Gemma that means applying the embedding scale here, because a supplied
  // row is used as-is by design - reproducing the baseline's INPUT is what
  // makes the comparison meaningful.
  const TensorInfo &embedding_table = tensor_at(model_state, model_state.tensor_index.token_embeddings);
  PromptEmbeddingOverrides overrides;
  for (std::size_t position = 0; position < token_ids.size(); position += 1) {
    std::vector<float> row = tensor_row_f32(model_state, embedding_table, token_ids[position]);
    if (gemma_path) {
      scale_vector_in_place(
        row,
        std::sqrt(static_cast<double>(model_state.model.embedding_length.value_or(1))));
    }
    overrides.rows.emplace(position, std::move(row));
  }

  SessionState supplied_session;
  FloatBatch supplied_logits;
  if (gemma_path) {
    execute_gemma_prefill_batch(model_state, supplied_session, token_ids, nullptr, &supplied_logits, &overrides);
  } else {
    execute_llama_prefill_batch(model_state, supplied_session, token_ids, nullptr, &supplied_logits, &overrides);
  }

  double worst_difference = 0.0;
  std::size_t compared = 0;
  bool shape_matches = baseline_logits.size() == supplied_logits.size();
  if (shape_matches) {
    for (std::size_t position = 0; position < baseline_logits.size(); position += 1) {
      if (baseline_logits[position].size() != supplied_logits[position].size()) {
        shape_matches = false;
        break;
      }
      for (std::size_t index = 0; index < baseline_logits[position].size(); index += 1) {
        const double difference = std::fabs(
          static_cast<double>(baseline_logits[position][index])
          - static_cast<double>(supplied_logits[position][index]));
        worst_difference = std::max(worst_difference, difference);
        compared += 1;
      }
    }
  }
  // Bit-identical is the bar. The two runs execute the same kernels on the same
  // values in the same order, so anything above zero means the supplied row
  // took a different path, and "close enough" would hide exactly that.
  const bool identical = shape_matches && worst_difference == 0.0;
  return JsonObject {
    {"ok", JsonValue(identical)},
    {"positions", JsonValue(static_cast<double>(token_ids.size()))},
    {"logitsCompared", JsonValue(static_cast<double>(compared))},
    {"maxAbsoluteDifference", JsonValue(worst_difference)},
    {"shapeMatches", JsonValue(shape_matches)},
  };
}

// ---- Vision tower forward pass (Phase 2, second half) ----------------------
//
// Takes an already-normalized RGB buffer and returns one language-model-width
// row per patch, ready to be spliced in as PromptEmbeddingOverrides.
//
// Image decoding and normalization deliberately live OUTSIDE this: they are
// Phase 3, they differ per model family, and keeping them out means this
// function is pure maths over a pixel buffer and can be diffed against a
// reference without arguing about resampling.
//
// UNVERIFIED until diffed against a reference implementation. The block order
// below is a reading of the tensor set, not a checked fact - in particular this
// file has no `ln2`, so whether the FFN sees a pre-norm is an open question
// (see .agent/vision/01_native_vision_plan.md). Do not wire this into a chat
// turn before the diff passes.
// Reuses the runner's existing layer_norm, add_bias_in_place and gelu_tanh.
// CLIP/SigLIP towers normalise with LayerNorm (mean-subtracting, biased);
// Gemma's inline tower uses RMSNorm and carries no bias. Confusing the two
// shifts every activation by the mean and yields fluent nonsense, not an error.
// Whichever model state actually owns the vision tensors: a separately-loaded
// mmproj when there is one, otherwise the model itself (Gemma 3n packs them
// inline). Everything downstream reads through this, so neither the tower nor
// its callers ever branch on how the weights arrived.
static const LoadedModelState &vision_state_of(const LoadedModelState &state) {
  return state.projector_model ? *state.projector_model : state;
}

static FloatBatch run_vision_tower(
  const LoadedModelState &model_state,
  const std::vector<float> &pixels,
  std::uint64_t width,
  std::uint64_t height) {
  const LoadedModelState &state = vision_state_of(model_state);
  const VisionTensorIndex &vision = state.vision_index;
  if (!vision.ready) {
    throw WorkerError("NATIVE_VISION_UNAVAILABLE", "This model does not carry a usable vision tower.");
  }
  const std::uint64_t patch = vision.patch_size;
  const std::uint64_t channels = vision.channels;
  if (patch == 0 || width % patch != 0 || height % patch != 0) {
    throw WorkerError(
      "NATIVE_VISION_BAD_IMAGE",
      "Image dimensions must be whole multiples of the patch size (" + std::to_string(patch) + ").");
  }
  if (pixels.size() != width * height * channels) {
    throw WorkerError(
      "NATIVE_VISION_BAD_IMAGE",
      "Pixel buffer is " + std::to_string(pixels.size()) + " values; expected "
        + std::to_string(width * height * channels) + ".");
  }

  const std::uint64_t grid_x = width / patch;
  const std::uint64_t grid_y = height / patch;
  std::uint64_t patch_count = grid_x * grid_y;
  const std::uint64_t hidden = vision.hidden_size;
  const std::uint64_t patch_values = patch * patch * channels;

  // Patch embedding. The conv kernel is [patch, patch, channels, hidden], which
  // as a matrix is hidden rows of patch*patch*channels - so patchifying turns
  // the convolution into the matvec the runner already does well.
  const TensorInfo &patch_weight = tensor_at(state, vision.patch_embedding);
  const MatrixShape patch_shape { patch_values, hidden };
  FloatBatch rows;
  rows.reserve(patch_count);
  for (std::uint64_t py = 0; py < grid_y; py += 1) {
    for (std::uint64_t px = 0; px < grid_x; px += 1) {
      std::vector<float> tile;
      tile.reserve(patch_values);
      // Row-major within the tile, channel-last, matching how the kernel's
      // leading dimensions are laid out.
      for (std::uint64_t y = 0; y < patch; y += 1) {
        for (std::uint64_t x = 0; x < patch; x += 1) {
          const std::uint64_t source_y = py * patch + y;
          const std::uint64_t source_x = px * patch + x;
          const std::uint64_t base = (source_y * width + source_x) * channels;
          for (std::uint64_t c = 0; c < channels; c += 1) {
            tile.push_back(pixels[base + c]);
          }
        }
      }
      rows.push_back(matrix_vector_mul_tensor(state, patch_weight, patch_shape, tile, "vision patch embedding"));
    }
  }

  // Position embeddings, one row per patch position.
  if (vision.position_embedding != k_missing_tensor_index) {
    const TensorInfo &position_weight = tensor_at(state, vision.position_embedding);
    for (std::uint64_t index = 0; index < patch_count; index += 1) {
      const std::vector<float> position_row = tensor_row_f32(state, position_weight, index);
      const std::size_t limit = std::min<std::size_t>(rows[index].size(), position_row.size());
      for (std::size_t value = 0; value < limit; value += 1) {
        rows[index][value] += position_row[value];
      }
    }
  }

  const std::uint64_t head_count = vision.head_count;
  const std::uint64_t head_dim = vision.head_dim;
  const float attention_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  const MatrixShape square { hidden, hidden };

  const VisionFamilySpec &family = vision.family;
  // One helper closes over the family so every norm site reads the same way and
  // no call site has to remember which kind this model uses.
  const auto normalize = [&](const std::vector<float> &input,
                             std::size_t weight_index,
                             std::size_t bias_index) {
    const std::vector<float> weight = tensor_vector_f32(state, tensor_at(state, weight_index));
    if (!family.layer_norm) {
      return rms_norm(input, weight, family.norm_epsilon);
    }
    std::vector<float> bias;
    const std::vector<float> *bias_pointer = nullptr;
    if (bias_index != k_missing_tensor_index) {
      bias = tensor_vector_f32(state, tensor_at(state, bias_index));
      bias_pointer = &bias;
    }
    return layer_norm(input, weight, bias_pointer, family.norm_epsilon);
  };
  const auto maybe_bias = [&](std::vector<float> &values, std::size_t bias_index) {
    if (bias_index == k_missing_tensor_index) return;
    add_bias_in_place(values, tensor_vector_f32(state, tensor_at(state, bias_index)), "vision projection");
  };

  for (const VisionLayerTensorIndex &layer : vision.layers) {
    FloatBatch queries(patch_count);
    FloatBatch keys(patch_count);
    FloatBatch values(patch_count);
    for (std::uint64_t index = 0; index < patch_count; index += 1) {
      const std::vector<float> normed = normalize(rows[index], layer.ln1, layer.ln1_bias);
      queries[index] = matrix_vector_mul_tensor(state, tensor_at(state, layer.attn_q), square, normed, "vision q");
      keys[index] = matrix_vector_mul_tensor(state, tensor_at(state, layer.attn_k), square, normed, "vision k");
      values[index] = matrix_vector_mul_tensor(state, tensor_at(state, layer.attn_v), square, normed, "vision v");
      maybe_bias(queries[index], layer.attn_q_bias);
      maybe_bias(keys[index], layer.attn_k_bias);
      maybe_bias(values[index], layer.attn_v_bias);
      // Per-head query/key norms, the Gemma detail a plain ViT does not have.
      if (layer.attn_q_norm != k_missing_tensor_index) {
        queries[index] = rms_norm_heads(
          queries[index], tensor_vector_f32(state, tensor_at(state, layer.attn_q_norm)),
          head_count, head_dim, "vision q norm");
      }
      if (layer.attn_k_norm != k_missing_tensor_index) {
        keys[index] = rms_norm_heads(
          keys[index], tensor_vector_f32(state, tensor_at(state, layer.attn_k_norm)),
          head_count, head_dim, "vision k norm");
      }
    }

    // Bidirectional attention: every patch sees every other patch. No causal
    // mask and no KV cache - an image is not a sequence being generated.
    FloatBatch attended(patch_count, std::vector<float>(hidden, 0.0f));
    for (std::uint64_t query_index = 0; query_index < patch_count; query_index += 1) {
      for (std::uint64_t head = 0; head < head_count; head += 1) {
        const std::uint64_t offset = head * head_dim;
        std::vector<float> weights(patch_count, 0.0f);
        for (std::uint64_t key_index = 0; key_index < patch_count; key_index += 1) {
          float dot = 0.0f;
          for (std::uint64_t d = 0; d < head_dim; d += 1) {
            dot += queries[query_index][offset + d] * keys[key_index][offset + d];
          }
          weights[key_index] = dot * attention_scale;
        }
        softmax_in_place(weights);
        for (std::uint64_t key_index = 0; key_index < patch_count; key_index += 1) {
          const float weight = weights[key_index];
          if (weight == 0.0f) continue;
          for (std::uint64_t d = 0; d < head_dim; d += 1) {
            attended[query_index][offset + d] += weight * values[key_index][offset + d];
          }
        }
      }
    }

    for (std::uint64_t index = 0; index < patch_count; index += 1) {
      std::vector<float> projected = matrix_vector_mul_tensor(
        state, tensor_at(state, layer.attn_out), square, attended[index], "vision attn out");
      maybe_bias(projected, layer.attn_out_bias);
      if (layer.attn_post_norm != k_missing_tensor_index) {
        projected = normalize(projected, layer.attn_post_norm, k_missing_tensor_index);
      }
      for (std::uint64_t value = 0; value < hidden; value += 1) {
        rows[index][value] += projected[value];
      }
    }

    // Gated FFN, the same silu(gate) * up shape the language layers use.
    // Roles come from SHAPES, not names.
    //
    // gemma-3's mmproj stores them the other way round - llama.cpp prints
    // "ffn up/down are swapped" when loading it - so `ffn_up` there is
    // [4304x1152], i.e. intermediate -> hidden, which is the DOWN projection.
    // Trusting the name would multiply against the wrong stride and, worse,
    // pair each projection with the wrong bias. A GGUF tensor's dims[0] is its
    // input width, so the projection that CONSUMES hidden is the up one
    // whatever it is called.
    std::size_t up_index = layer.ffn_up;
    std::size_t up_bias_index = layer.ffn_up_bias;
    std::size_t down_index = layer.ffn_down;
    std::size_t down_bias_index = layer.ffn_down_bias;
    {
      const TensorInfo &named_up = tensor_at(state, layer.ffn_up);
      const bool named_up_consumes_hidden = named_up.dimensions.size() >= 2
        && named_up.dimensions[0] == hidden;
      if (!named_up_consumes_hidden) {
        std::swap(up_index, down_index);
        std::swap(up_bias_index, down_bias_index);
      }
    }
    const TensorInfo &up_tensor = tensor_at(state, up_index);
    const TensorInfo &down_tensor = tensor_at(state, down_index);
    const TensorInfo &gate_tensor = family.gated_ffn ? tensor_at(state, layer.ffn_gate) : up_tensor;
    const std::uint64_t intermediate = up_tensor.dimensions.size() >= 2 ? up_tensor.dimensions[1] : 0;
    if (intermediate == 0 || up_tensor.dimensions[0] != hidden) {
      throw WorkerError(
        "NATIVE_VISION_UNSUPPORTED",
        "The vision MLP projections do not resolve to a hidden -> intermediate -> hidden pair.");
    }
    const MatrixShape wide { hidden, intermediate };
    const MatrixShape narrow { intermediate, hidden };
    for (std::uint64_t index = 0; index < patch_count; index += 1) {
      // ln2 is the FFN's pre-norm where a family has one. Gemma's inline tower
      // does not, and feeds the post-attention residual straight in.
      const std::vector<float> ffn_input = layer.ln2 != k_missing_tensor_index
        ? normalize(rows[index], layer.ln2, layer.ln2_bias)
        : rows[index];
      std::vector<float> activated(intermediate, 0.0f);
      const std::vector<float> up = matrix_vector_mul_tensor(state, up_tensor, wide, ffn_input, "vision ffn up");
      if (family.gated_ffn) {
        const std::vector<float> gate = matrix_vector_mul_tensor(state, gate_tensor, wide, ffn_input, "vision ffn gate");
        for (std::uint64_t value = 0; value < intermediate; value += 1) {
          activated[value] = silu(gate[value]) * up[value];
        }
      } else {
        // Plain MLP: one projection, one activation. No gate exists to read.
        std::vector<float> biased = up;
        maybe_bias(biased, up_bias_index);
        for (std::uint64_t value = 0; value < intermediate; value += 1) {
          activated[value] = family.gelu_activation ? gelu_tanh(biased[value]) : silu(biased[value]);
        }
      }
      std::vector<float> down = matrix_vector_mul_tensor(state, down_tensor, narrow, activated, "vision ffn down");
      maybe_bias(down, down_bias_index);
      if (layer.ffn_post_norm != k_missing_tensor_index) {
        down = normalize(down, layer.ffn_post_norm, k_missing_tensor_index);
      }
      for (std::uint64_t value = 0; value < hidden; value += 1) {
        rows[index][value] += down[value];
      }
    }
  }

  // ---- pool the patch grid ----
  //
  // The language model was trained to see a FIXED number of image positions
  // (256 for gemma-3), not one per patch. Handing it 4096 is a different input
  // entirely - it would answer, fluently and wrongly. Pooling is average over
  // pool_factor x pool_factor cells of the grid, which is what collapses
  // 64x64 patches to 16x16.
  if (family.pool_factor > 1) {
    const std::uint64_t factor = family.pool_factor;
    if (grid_x % factor != 0 || grid_y % factor != 0) {
      throw WorkerError(
        "NATIVE_VISION_BAD_IMAGE",
        "The patch grid (" + std::to_string(grid_x) + "x" + std::to_string(grid_y)
          + ") does not divide by this projector's pooling factor (" + std::to_string(factor) + ").");
    }
    const std::uint64_t pooled_x = grid_x / factor;
    const std::uint64_t pooled_y = grid_y / factor;
    FloatBatch pooled;
    pooled.reserve(pooled_x * pooled_y);
    const float inverse = 1.0f / static_cast<float>(factor * factor);
    for (std::uint64_t cy = 0; cy < pooled_y; cy += 1) {
      for (std::uint64_t cx = 0; cx < pooled_x; cx += 1) {
        std::vector<float> cell(hidden, 0.0f);
        for (std::uint64_t dy = 0; dy < factor; dy += 1) {
          for (std::uint64_t dx = 0; dx < factor; dx += 1) {
            const std::vector<float> &source = rows[(cy * factor + dy) * grid_x + (cx * factor + dx)];
            for (std::uint64_t value = 0; value < hidden; value += 1) {
              cell[value] += source[value];
            }
          }
        }
        for (std::uint64_t value = 0; value < hidden; value += 1) {
          cell[value] *= inverse;
        }
        pooled.push_back(std::move(cell));
      }
    }
    rows = std::move(pooled);
    patch_count = pooled_x * pooled_y;
  }

  // Project each patch into the language model's embedding width. These rows
  // are what a prompt splice consumes.
  const TensorInfo &projection = tensor_at(state, vision.input_projection);
  // matrix_vector_mul_tensor consumes {row_width = input, row_count = output}.
  // Exporters disagree on the projector's orientation, so both are supported
  // rather than assumed - feeding the wrong one multiplies against the wrong
  // stride and produces plausible-looking noise instead of an error.
  const MatrixShape projection_shape { vision.hidden_size, vision.language_hidden_size };
  FloatBatch projected;
  projected.reserve(patch_count);
  for (std::uint64_t index = 0; index < patch_count; index += 1) {
    // A soft-embedding norm sits between the tower and the projector on
    // gemma-3's mmproj. Absent elsewhere, so it is applied only when present.
    std::vector<float> tower_output = rows[index];
    if (vision.soft_embedding_norm != k_missing_tensor_index) {
      tower_output = rms_norm(
        tower_output,
        tensor_vector_f32(state, tensor_at(state, vision.soft_embedding_norm)),
        family.norm_epsilon);
    }
    if (vision.projection_is_vision_major) {
      projected.push_back(
        matrix_vector_mul_tensor(state, projection, projection_shape, tower_output, "vision projector"));
      continue;
    }
    // Stored [language, vision]: the tensor holds `vision` rows of `language`
    // width, so the product is an accumulation of rows weighted by the input
    // rather than a row-wise dot. Same maths, transposed access.
    std::vector<float> out(vision.language_hidden_size, 0.0f);
    for (std::uint64_t source = 0; source < vision.hidden_size; source += 1) {
      const float weight = tower_output[source];
      if (weight == 0.0f) continue;
      const std::vector<float> row = tensor_row_f32(state, projection, source);
      const std::size_t limit = std::min<std::size_t>(out.size(), row.size());
      for (std::size_t target = 0; target < limit; target += 1) {
        out[target] += weight * row[target];
      }
    }
    projected.push_back(std::move(out));
  }
  return projected;
}

// Phase 2, first half: report what vision tower this model actually carries.
// Loading is worth verifying on its own - a tower indexed from the wrong names
// produces confident nonsense later, and that is much harder to see than a
// missing tensor here.
static JsonObject handle_vision_probe(WorkerState &state, const JsonObject &params) {
  (void)params;
  if (state.loaded_model == nullptr) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Load a model before probing its vision tower.");
  }
  const VisionTensorIndex &vision = vision_state_of(*state.loaded_model).vision_index;
  JsonArray missing;
  for (const std::string &name : vision.missing) {
    missing.push_back(JsonValue(name));
  }
  return JsonObject {
    {"ready", JsonValue(vision.ready)},
    {"blocks", JsonValue(static_cast<double>(vision.layers.size()))},
    {"patchSize", JsonValue(static_cast<double>(vision.patch_size))},
    {"channels", JsonValue(static_cast<double>(vision.channels))},
    {"visionHiddenSize", JsonValue(static_cast<double>(vision.hidden_size))},
    {"languageHiddenSize", JsonValue(static_cast<double>(vision.language_hidden_size))},
    {"headCount", JsonValue(static_cast<double>(vision.head_count))},
    {"headDim", JsonValue(static_cast<double>(vision.head_dim))},
    {"family", JsonValue(JsonObject {
      {"label", JsonValue(vision.family.label)},
      {"layerNorm", JsonValue(vision.family.layer_norm)},
      {"gatedFfn", JsonValue(vision.family.gated_ffn)},
      {"qkNorms", JsonValue(vision.family.qk_norms)},
      {"postNorms", JsonValue(vision.family.post_norms)},
      {"projectionBiases", JsonValue(vision.family.projection_biases)},
      {"softEmbeddingNorm", JsonValue(vision.family.soft_embedding_norm)},
      {"geluActivation", JsonValue(vision.family.gelu_activation)},
      {"normEpsilon", JsonValue(vision.family.norm_epsilon)},
      {"imageSize", JsonValue(static_cast<double>(vision.family.image_size))},
      {"poolFactor", JsonValue(static_cast<double>(vision.family.pool_factor))},
      {"projectorType", JsonValue(vision.family.projector_type)},
    })},
    {"hasPositionEmbedding", JsonValue(vision.position_embedding != k_missing_tensor_index)},
    {"separateProjector", JsonValue(static_cast<bool>(state.loaded_model->projector_model))},
    {"missing", JsonValue(missing)},
  };
}

// Runs the tower over a caller-supplied pixel buffer and returns the projected
// rows, plus per-row checksums. The checksums are what a reference diff
// compares first: if they disagree, dumping full rows is worth the bandwidth;
// if they agree, nothing else needs sending.
static JsonObject handle_vision_encode(WorkerState &state, const JsonObject &params) {
  if (state.loaded_model == nullptr) {
    throw WorkerError("NATIVE_WORKER_ERROR", "Load a model before encoding an image.");
  }
  const std::uint64_t width = static_cast<std::uint64_t>(json_number_or_default(params, "width", 0));
  const std::uint64_t height = static_cast<std::uint64_t>(json_number_or_default(params, "height", 0));
  const JsonValue *pixels_value = find_object_value(params, "pixels");
  std::vector<float> pixels;
  if (pixels_value != nullptr && std::holds_alternative<JsonArray>(pixels_value->value)) {
    const JsonArray &array = std::get<JsonArray>(pixels_value->value);
    pixels.reserve(array.size());
    for (const JsonValue &entry : array) {
      pixels.push_back(static_cast<float>(std::holds_alternative<double>(entry.value)
        ? std::get<double>(entry.value)
        : 0.0));
    }
  }
  const bool include_rows = json_bool_or_default(params, "includeRows", false);

  const auto started_at = std::chrono::steady_clock::now();
  const FloatBatch rows = run_vision_tower(*state.loaded_model, pixels, width, height);
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started_at).count();

  JsonArray checksums;
  JsonArray first_row;
  for (std::size_t index = 0; index < rows.size(); index += 1) {
    double sum = 0.0;
    for (float value : rows[index]) {
      sum += static_cast<double>(value);
    }
    checksums.push_back(JsonValue(sum));
  }
  if (include_rows && !rows.empty()) {
    for (float value : rows.front()) {
      first_row.push_back(JsonValue(static_cast<double>(value)));
    }
  }
  return JsonObject {
    {"patches", JsonValue(static_cast<double>(rows.size()))},
    {"width", JsonValue(static_cast<double>(rows.empty() ? 0 : rows.front().size()))},
    {"elapsedMs", JsonValue(elapsed_ms)},
    {"rowSums", JsonValue(checksums)},
    {"firstRow", JsonValue(first_row)},
    {"verified", JsonValue(false)},
    {"note", JsonValue(std::string(
      "Tower output is UNVERIFIED: the block order is a reading of the tensor set, "
      "not a checked fact. Diff against a reference before trusting it."))},
  };
}

static JsonObject dispatch(WorkerState &state, const std::string &method, const JsonObject &params) {
  if (method == "embedding_prompt_selftest") {
    return handle_embedding_prompt_selftest(state, params);
  }
  if (method == "vision_probe") {
    return handle_vision_probe(state, params);
  }
  if (method == "vision_encode") {
    return handle_vision_encode(state, params);
  }
  if (method == "handshake") {
    return build_handshake_result(state);
  }
  if (method == "probe") {
    return handle_probe();
  }
  if (method == "load_model") {
    return handle_load_model(state, params);
  }
  if (method == "unload_model") {
    return handle_unload_model(state);
  }
  if (method == "session_open") {
    return handle_session_open(state, params);
  }
  if (method == "session_clone") {
    return handle_session_clone(state, params);
  }
  if (method == "session_close") {
    return handle_session_close(state, params);
  }
  if (method == "prefill") {
    return handle_prefill(state, params);
  }
  if (method == "generate" || method == "decode") {
    return handle_generate_or_decode(state, params);
  }
  if (method == "decode_batch") {
    return handle_decode_batch(state, params);
  }
  if (method == "cancel") {
    return handle_cancel(state, params);
  }
  if (method == "metrics") {
    return handle_metrics(state);
  }
  if (method == "tensor_benchmark") {
    return handle_tensor_benchmark(state, params);
  }
  if (method == "shutdown") {
    return handle_shutdown(state);
  }
  throw WorkerError("NATIVE_WORKER_ERROR", "Unsupported native worker method.");
}

// EnhancementRun_04 Phase 2: guarded, because the step loop makes more than one
// thread capable of reaching stdout. A line is written whole or not at all --
// two interleaved half-lines would corrupt the protocol in a way that looks
// like a parser bug on the JS side.
//
// The mutex costs nothing on the synchronous path (uncontended) and is what
// lets streaming token events from a decoding sequence coexist with responses.
static void write_line(const JsonValue &value) {
  static std::mutex output_mutex;
  const std::string line = stringify_json(value);
  std::lock_guard<std::mutex> guard(output_mutex);
  std::cout << line << '\n';
  std::cout.flush();
}

// EnhancementRun_04 Phase 2 -- one request, handled identically by both loops.
//
// Extracted verbatim from main()'s while body so the synchronous path and the
// step-loop path cannot drift apart. That is what makes the phase's gate --
// byte-identical output with the flag on -- true by construction rather than by
// testing: there is only one implementation.
static void process_request_line(WorkerState &state, const std::string &line) {
  JsonValue id;
  try {
    JsonValue request = JsonParser(line).parse();
    if (!request.is_object()) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Native worker requests must be JSON objects.");
    }
    const JsonObject &request_object = request.as_object();
    if (const JsonValue *value = find_object_value(request_object, "id"); value != nullptr) {
      id = *value;
    }
    const std::string method = json_string_or_default(request_object, "method");
    if (method.empty()) {
      throw WorkerError("NATIVE_WORKER_ERROR", "Native worker requests must include a method.");
    }
    const JsonObject *params = find_object_field(request_object, "params");
    JsonObject result = dispatch(state, method, params == nullptr ? JsonObject {} : *params);
    write_line(success_response(id, std::move(result)));
  } catch (const WorkerError &error) {
    write_line(error_response(id, error));
  } catch (const std::exception &error) {
    write_line(error_response(id, WorkerError("NATIVE_WORKER_ERROR", error.what())));
  }
}

}  // namespace lizard_native

int main(int argc, char **argv) {
  using namespace lizard_native;

  for (int index = 1; index < argc; index += 1) {
    if (std::string(argv[index]) == "--cpu-selftest") {
      // Reaching main and executing a small arithmetic loop is the startup
      // self-test: an incompatible optimized binary terminates with illegal
      // instruction before returning success, so the JS launcher can retry
      // the portable baseline without loading a model.
      volatile float accumulator = 0.0f;
      for (int value = 0; value < 256; value += 1) {
        accumulator += static_cast<float>(value) * 0.5f;
      }
      if (!(accumulator > 0.0f)) {
        return 1;
      }
      if (!ggml_cpu_backend_selection_error().empty()) {
        return 1;
      }
      if (ggml_cpu_backend_selected()) {
        std::string ggml_error;
        if (!ggml_cpu_backend_selftest(&ggml_error)) {
          return 1;
        }
      }
#ifdef LIZARD_NATIVE_AVX2
      // The optimized worker selection gate now proves the highest-value
      // packed kernel too, not merely that the process can execute one AVX2
      // instruction. Any reconstruction, scale, or offset regression makes
      // the launcher fall through to the portable binary.
      if (!q6_k_q8_k_kernel_selftest()) {
        return 1;
      }
      if (!q4_k_q8_k_kernel_selftest()) {
        return 1;
      }
#endif
      return 0;
    }
    if (std::string(argv[index]) == "--ggml-cpu-selftest") {
      if (!ggml_cpu_backend_compiled()) {
        return 2;
      }
      std::string ggml_error;
      if (!ggml_cpu_backend_selftest(&ggml_error)) {
        std::cerr << ggml_error << '\n';
        return 1;
      }
      return 0;
    }
    if (std::string(argv[index]) == "--d3d12-selftest") {
      return d3d12_matmul_selftest();
    }
    // EnhancementRun_04 Phase 3 gate. Runs without a model on purpose: the
    // block pool is pure bookkeeping, so its correctness must be provable on
    // any machine, including CI boxes with no GGUF.
    if (std::string(argv[index]) == "--kv-blocks-selftest") {
      return kv_blocks_selftest();
    }
    if (std::string(argv[index]) == "--prefix-cache-selftest") {
      return prefix_cache_selftest();
    }
    if (std::string(argv[index]) == "--probe-hardware") {
      return run_hardware_probe_main();
    }
  }

  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  WorkerState state;

  // EnhancementRun_04 Phase 2. Opt-in, default off: until the gate passes, the
  // synchronous loop below remains the shipping path and reverting is one env
  // var. Both paths run the same process_request_line, so the only difference
  // is which thread reads stdin.
  if (engine_loop_enabled()) {
    EngineLoopHooks hooks;
    hooks.handle_line = [&state](const std::string &request_line) {
      process_request_line(state, request_line);
    };
    hooks.shutdown_requested = [&state] { return state.shutdown_requested; };
    return run_engine_loop(hooks);
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    if (trim(line).empty()) {
      continue;
    }
    process_request_line(state, line);
    if (state.shutdown_requested) {
      break;
    }
  }

  return 0;
}
