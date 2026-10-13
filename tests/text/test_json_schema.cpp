#include "text/grammar.h"
#include "../../third_party/xgrammar/cpp/compiled_grammar_impl.h"
#include "../../third_party/xgrammar/cpp/grammar_builder.h"
#include "../../third_party/xgrammar/cpp/json_string_grammar.h"
#include "../../third_party/xgrammar/cpp/testing.h"
#include <nlohmann/json.hpp>
#include <xgrammar/xgrammar.h>
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
    auto mixed_length =
        Constraint::json_schema(R"({"type":"string","minLength":2,"maxLength":3})");
    for (const std::string value :
         std::vector<std::string>{"a\n", "\"你", "\\😀x"})
        check(accepts(compiler, mixed_length, Json(value).dump()),
              "mixed raw, escaped, or non-BMP scalar length mismatch");
    for (const std::string value : std::vector<std::string>{"a", "a\n你😀"})
        check(!accepts(compiler, mixed_length, Json(value).dump()),
              "mixed scalar length bound was ignored");
    auto pattern = Constraint::json_schema(R"({"type":"string","pattern":"p"})");
    check(accepts(compiler, pattern, "\"apple\""), "pattern must use search semantics");
    check(!accepts(compiler, pattern, "\"absent\""), "pattern admitted a nonmatching string");
    const auto dot = Constraint::json_schema(R"({"type":"string","pattern":"^a.b$"})");
    for (const std::string separator : {"\n", "\r", "\u2028", "\u2029"})
        check(!accepts(compiler, dot, Json("a" + separator + "b").dump()),
              "ECMAScript dot admitted a line terminator");
}

void counted_repetition_contract() {
    const std::vector<std::string> vocabulary{"a", "aaaa", "\\n", "你", "😀", "\"", ""};
    xgrammar::TokenizerInfo tokenizer(vocabulary, xgrammar::VocabType::RAW, std::nullopt,
                                      std::vector<std::int32_t>{6});
    xgrammar::GrammarCompiler compiler(tokenizer, 1, false);
    const auto grammar =
        xgrammar::Grammar::FromJSONSchema(R"({"type":"string","minLength":12,"maxLength":20})",
                                          false, std::nullopt,
                                          std::pair<std::string, std::string>{",", ":"}, false);
    const auto compiled = compiler.CompileGrammar(grammar);
    const auto optimized = compiled.GetGrammar().ToString();
    if (optimized.find("{12, 20}") == std::string::npos)
        std::cerr << "unoptimized bounded string grammar:\n"
                  << grammar.ToString() << "\noptimized bounded string grammar:\n"
                  << optimized << '\n';
    check(optimized.find("{12, 20}") != std::string::npos,
          "JSON scalar length repetition was expanded instead of counted");
    std::size_t repeat_fast_path_masks = 0;
    for (const auto& [state, mask] : compiled.ImplPtr()->adaptive_token_mask_cache) {
        (void)state;
        if (!mask.repeat_interior_bitsets.empty() || !mask.repeat_exit_tokens.empty())
            ++repeat_fast_path_masks;
    }
    if (repeat_fast_path_masks == 0)
        std::cerr << "counted string grammar has no repeat fast-path token masks:\n"
                  << optimized << "\nFSMs:\n"
                  << xgrammar::_PrintGrammarFSMs(compiled.GetGrammar()) << '\n';
    check(repeat_fast_path_masks > 0,
          "counted string grammar did not build repeat fast-path token masks");

    const auto ordinary_choice =
        compiler.CompileGrammar(R"(root ::= ("a" | "bb"){12,20} "!")").GetGrammar().ToString();
    check(ordinary_choice.find("{12, 20}") == std::string::npos,
          "arbitrary variable-width choice was treated as one counted scalar");

    const auto find_scalar_rule = [](const xgrammar::Grammar& grammar) {
        int32_t scalar_rule_id = -1;
        for (int32_t rule_id = 0; rule_id < grammar->NumRules(); ++rule_id) {
            if (!xgrammar::IsJSONEncodedUnicodeScalar(
                    grammar, grammar->GetRule(rule_id).body_expr_id))
                continue;
            check(scalar_rule_id == -1, "bounded string grammar has multiple scalar rules");
            scalar_rule_id = rule_id;
        }
        check(scalar_rule_id != -1, "bounded string grammar has no recognized scalar rule");
        return scalar_rule_id;
    };

    auto capture_grammar = xgrammar::JSONStringLengthAndClose(2, 3);
    xgrammar::GrammarBuilder capture_builder(capture_grammar);
    capture_builder.UpdateCaptureName(find_scalar_rule(capture_grammar), "scalar");
    const auto capture_compiled =
        compiler.CompileGrammar(capture_builder.Get(capture_grammar->GetRootRuleId()));
    xgrammar::GrammarMatcher capture_matcher(capture_compiled);
    check(capture_matcher.AcceptString("ab\""), "captured scalar repetition rejected valid input");
    check(
        capture_matcher.GetCaptures() ==
            std::vector<std::pair<std::string, std::string>>{{"scalar", "a"}, {"scalar", "b"}},
        "captured scalar repetition lost completion events"
    );

    auto lookahead_grammar = xgrammar::JSONStringLengthAndClose(2, 3);
    xgrammar::GrammarBuilder lookahead_builder(lookahead_grammar);
    const auto lookahead_rule_id = find_scalar_rule(lookahead_grammar);
    lookahead_builder.UpdateLookaheadAssertion(
        lookahead_rule_id, lookahead_builder.AddByteString("\"")
    );
    const auto lookahead_compiled =
        compiler.CompileGrammar(lookahead_builder.Get(lookahead_grammar->GetRootRuleId()));
    bool kept_lookahead = false;
    for (int32_t rule_id = 0; rule_id < lookahead_compiled.GetGrammar()->NumRules(); ++rule_id)
        kept_lookahead =
            kept_lookahead ||
            lookahead_compiled.GetGrammar()->GetRule(rule_id).lookahead_assertion_id != -1;
    check(kept_lookahead, "scalar repetition erased lookahead metadata");
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
            counted_repetition_contract();
            std::cout << "JSON/schema CPU contracts passed\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
