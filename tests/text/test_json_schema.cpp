#include "text/grammar.h"
#include "../../third_party/xgrammar/cpp/compiled_grammar_impl.h"
#include "../../third_party/xgrammar/cpp/grammar_builder.h"
#include <xgrammar/xgrammar.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using Json       = nlohmann::ordered_json;
using Constraint = ninfer::OutputConstraint;

ninfer::text::GrammarCompiler make_compiler() {
    std::vector<std::string> vocabulary;
    for (int i = 0; i < 256; ++i) vocabulary.emplace_back(1, static_cast<char>(i));
    vocabulary.emplace_back();
    return {std::move(vocabulary), {256}, 4 * 1024 * 1024};
}

bool accepts(ninfer::text::GrammarCompiler& compiler, const Constraint& constraint,
             const std::string& candidate) {
    auto matcher = compiler.compile(constraint, {}, {});
    std::vector<std::uint32_t> words(matcher->mask_words());
    const auto allows = [&](int token) {
        return matcher->masks({}, words) == 0 && (words[token / 32] & (1u << (token % 32)));
    };
    for (unsigned char byte : candidate) {
        if (!allows(byte)) return false;
        matcher->accept(byte);
        matcher->confirm();
    }
    return allows(256);
}

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void length_boundaries(ninfer::text::GrammarCompiler& compiler) {
    for (int upper : {0, 1, 127, 128, 129, 200}) {
        for (int lower : {0, upper}) {
            auto schema = Json{{"type", "string"}, {"minLength", lower}, {"maxLength", upper}};
            const auto constraint = Constraint::json_schema(schema.dump());
            for (int count : {std::max(0, upper - 1), upper, upper + 1}) {
                const bool expected = lower <= count && count <= upper;
                check(accepts(compiler, constraint, Json(std::string(count, 'a')).dump()) ==
                          expected,
                      "bounded JSON string accepted the wrong length");
            }
            if (upper == 0) continue;
            for (const auto& last :
                 std::vector<std::string>{"你", "😀", "\n", "\"", "\\", std::string(1, '\0')}) {
                const std::string prefix(upper - 1, 'a');
                check(accepts(compiler, constraint, Json(prefix + last).dump()),
                      "last JSON scalar did not fit its length budget");
                check(!accepts(compiler, constraint, Json(prefix + last + "a").dump()),
                      "JSON escape or UTF-8 exceeded its length budget");
            }
        }
    }
    const auto minimum = Constraint::json_schema(R"({"type":"string","minLength":129})");
    check(!accepts(compiler, minimum, Json(std::string(128, 'a')).dump()) &&
              accepts(compiler, minimum, Json(std::string(200, 'a')).dump()),
          "unbounded JSON length lost its minimum");

    const auto shared = Constraint::json_schema(R"({"type":"object","properties":{
        "a":{"type":"string","minLength":1,"maxLength":2},
        "b":{"type":"string","minLength":3,"maxLength":4}},
        "required":["a","b"],"additionalProperties":false})");
    check(accepts(compiler, shared, R"({"a":"你","b":"a\nb"})") &&
              !accepts(compiler, shared, R"({"a":"abc","b":"abc"})") &&
              !accepts(compiler, shared, R"({"a":"a","b":"ab"})"),
          "shared JSON scalar rules confused field length bounds");
}

void dense_length_masks() {
    std::vector<std::string> vocabulary{"\"", "a\"", "aa\"", "aaaa\"", "\"a"};
    for (char a = 'a'; a <= 'z'; ++a)
        for (char b = 'a'; b <= 'z'; ++b)
            for (char c = 'a'; c <= 'z'; ++c) vocabulary.push_back(std::string{a, b, c});
    const int eos = static_cast<int>(vocabulary.size());
    vocabulary.emplace_back();
    ninfer::text::GrammarCompiler compiler(vocabulary, {eos}, 8 * 1024 * 1024);
    const auto constraint =
        Constraint::json_schema(R"({"type":"string","minLength":127,"maxLength":129})");
    for (int consumed : {0, 126, 127, 128, 129}) {
        auto session = compiler.compile(constraint, {}, "\"" + std::string(consumed, 'a'));
        std::vector<std::uint32_t> words(session->mask_words());
        check(session->masks({}, words) == 0, "length mask reached a dead end");
        for (int token = 0; token <= eos; ++token) {
            bool expected = token >= 5 && token < eos && consumed + 3 <= 129;
            if (token < 4) {
                const int length = consumed + static_cast<int>(vocabulary[token].size()) - 1;
                expected         = length >= 127 && length <= 129;
            }
            check(bool(words[token / 32] & (1u << (token % 32))) == expected,
                  "dense JSON mask differs from independent length bounds");
        }
    }
}

void contracts(ninfer::text::GrammarCompiler& compiler) {
    auto object = Constraint::json_object();
    for (const std::string text : {"{}", R"({"x":[1,true,null,{"y":"你好"}]})"})
        check(accepts(compiler, object, text), "JSON object rejected valid content");
    for (const std::string text : {"[]", "1", "{}{}", R"({"x":})"})
        check(!accepts(compiler, object, text), "JSON object admitted invalid content");
    auto schema = Constraint::json_schema(
        R"({"type":"object","properties":{"description":{"const":"first"},"second":{"const":{"description":"second"}}},"required":["description","second"],"additionalProperties":false})");
    check(accepts(compiler, schema, R"({"description":"first","second":{"description":"second"}})"),
          "annotation-named properties were lost");
    check(!accepts(compiler, schema, R"({"description":"first","second":{"description":"first"}})"),
          "literal data was aliased in schema cache");
    auto length = Constraint::json_schema(R"({"type":"string","minLength":1,"maxLength":1})");
    for (const std::string value :
         std::vector<std::string>{"你", "😀", "\n", "\t", std::string(1, '\0'), "\"", "\\"})
        check(accepts(compiler, length, Json(value).dump()),
              "JSON character length/encoding mismatch");
    check(!accepts(compiler, length, Json("ab").dump()), "string length was ignored");
    const auto mixed = Constraint::json_schema(
        R"({"type":"string","minLength":2,"maxLength":3})");
    for (const auto& value : std::vector<std::string>{"a\n", "\"你", "\\😀x"})
        check(accepts(compiler, mixed, Json(value).dump()),
              "mixed raw, escaped or non-BMP scalar length mismatch");
    for (const auto& value : std::vector<std::string>{"a", "a\n你😀"})
        check(!accepts(compiler, mixed, Json(value).dump()),
              "mixed scalar length bound was ignored");
    auto pattern = Constraint::json_schema(R"({"type":"string","pattern":"p"})");
    check(accepts(compiler, pattern, "\"apple\""), "pattern must use search semantics");
    check(!accepts(compiler, pattern, "\"absent\""), "pattern admitted a nonmatching string");
    const auto dot = Constraint::json_schema(R"({"type":"string","pattern":"^a.b$"})");
    for (const std::string separator : {"\n", "\r", "\u2028", "\u2029"})
        check(!accepts(compiler, dot, Json("a" + separator + "b").dump()),
              "ECMAScript dot admitted a line terminator");
}

void counted_repetition_metadata() {
    const std::vector<std::string> vocabulary{"a", "aaaa", "\\n", "你", "😀", "\"", ""};
    xgrammar::TokenizerInfo tokenizer(vocabulary, xgrammar::VocabType::RAW, std::nullopt,
                                     std::vector<std::int32_t>{6});
    xgrammar::GrammarCompiler compiler(tokenizer, 1, false);
    const auto grammar = xgrammar::Grammar::FromJSONSchema(
        R"({"type":"string","minLength":12,"maxLength":20})", false, std::nullopt,
        std::pair<std::string, std::string>{",", ":"}, false);
    const auto compiled = compiler.CompileGrammar(grammar);
    check(compiled.GetGrammar().ToString().find("{12, 20}") != std::string::npos,
          "bounded JSON repetition was expanded instead of counted");
    std::size_t fast_masks = 0;
    for (const auto& [state, mask] : compiled.ImplPtr()->adaptive_token_mask_cache) {
        (void)state;
        if (!mask.repeat_interior_bitsets.empty() || !mask.repeat_exit_tokens.empty() ||
            !mask.repeat_interior_tokens.empty())
            ++fast_masks;
    }
    check(fast_masks > 0, "counted JSON repetition did not build its token-mask fast path");

    // Capture and lookahead metadata must survive scalar-rule optimization.
    const auto captured = compiler.CompileGrammar(
        "root ::= scalar{2,3} \"\\\"\"\nscalar[capture=\"scalar\"] ::= \"a\" | \"你\" | \"😀\"");
    xgrammar::GrammarMatcher matcher(captured);
    check(matcher.AcceptString("a你\""), "captured repetition rejected valid input");
    const std::vector<std::pair<std::string, std::string>> expected{
        {"scalar", "a"}, {"scalar", "你"}};
    check(matcher.GetCaptures() == expected, "counted repetition lost scalar captures");
    matcher.Rollback(1);
    check(matcher.GetCaptures().empty(), "rollback retained tentative scalar captures");
    check(matcher.AcceptString("😀a\"") &&
              matcher.GetCaptures() ==
                  std::vector<std::pair<std::string, std::string>>{
                      {"scalar", "😀"}, {"scalar", "a"}},
          "replay after rollback changed scalar capture bytes");

    const auto lookahead = compiler.CompileGrammar(
        "root ::= scalar{2,3} \"\\\"\"\nscalar ::= \"a\" | \"你\" (= \"\\\"\")");
    bool kept = false;
    for (std::int32_t rule = 0; rule < lookahead.GetGrammar()->NumRules(); ++rule)
        kept |= lookahead.GetGrammar()->GetRule(rule).lookahead_assertion_id != -1;
    check(kept, "counted repetition erased lookahead metadata");
}
} // namespace

int main(int argc, char** argv) {
    try {
        auto compiler = make_compiler();
        if (argc == 2 && std::string_view(argv[1]) == "--probe") {
            std::string line;
            while (std::getline(std::cin, line)) {
                Json response;
                try {
                    const auto input = Json::parse(line);
                    const auto constraint =
                        input.value("json_object", false)
                            ? Constraint::json_object()
                            : Constraint::json_schema(
                                  input.contains("schema_source")
                                      ? input["schema_source"].get<std::string>()
                                      : input.at("schema").dump());
                    auto session         = compiler.compile(constraint, {}, {});
                    response["accepted"] = Json::array();
                    for (const auto& text : input.at("candidates"))
                        response["accepted"].push_back(
                            accepts(compiler, constraint, text.get<std::string>()));
                } catch (const ninfer::RequestError& error) {
                    response = {{"error", error.what()},
                                {"kind", static_cast<int>(error.kind())},
                                {"pointer", error.pointer()}};
                } catch (const std::exception& error) { response = {{"error", error.what()}}; }
                std::cout << response.dump() << std::endl;
            }
        } else {
            contracts(compiler);
            length_boundaries(compiler);
            dense_length_masks();
            counted_repetition_metadata();
            std::cout << "JSON/schema CPU contracts passed\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
