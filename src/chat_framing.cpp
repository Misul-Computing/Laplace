#include "chat_framing.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace Laplace {
namespace {

// ---------------------------------------------------------------------------
// Template lexing: text runs, {{ expressions }}, and {% tags %} with the
// whitespace-control dash forms.
// ---------------------------------------------------------------------------

enum class SegmentKind { Text, Expr, Tag };

struct Segment {
    SegmentKind kind = SegmentKind::Text;
    std::string_view text;
};

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool lex_template(std::string_view source, std::vector<Segment>& segments) {
    size_t offset = 0;
    bool pending_left_strip = false;
    // A leading dash on {{- or {%- strips the tail of the text that was
    // already emitted; trim it off (and drop the segment if it empties).
    const auto strip_previous_tail = [&segments]() {
        if (segments.empty() || segments.back().kind != SegmentKind::Text) return;
        std::string_view text = segments.back().text;
        size_t end = text.size();
        while (end > 0 && is_space(text[end - 1])) --end;
        if (end == 0) {
            segments.pop_back();
        } else if (end != text.size()) {
            segments.back().text = text.substr(0, end);
        }
    };
    while (offset < source.size()) {
        // Next "{{" or "{%" from offset.
        size_t next = std::string_view::npos;
        SegmentKind kind = SegmentKind::Text;
        for (size_t scan = offset; scan + 1 < source.size(); ++scan) {
            if (source[scan] != '{' || (source[scan + 1] != '{' && source[scan + 1] != '%'))
                continue;
            next = scan;
            kind = source[scan + 1] == '{' ? SegmentKind::Expr : SegmentKind::Tag;
            break;
        }
        if (next == std::string_view::npos) {
            std::string_view text = source.substr(offset);
            if (pending_left_strip) {
                size_t begin = 0;
                while (begin < text.size() && is_space(text[begin])) ++begin;
                text = text.substr(begin);
            }
            if (!text.empty()) segments.push_back({SegmentKind::Text, text});
            return true;
        }
        if (next > offset) {
            std::string_view text = source.substr(offset, next - offset);
            if (pending_left_strip) {
                size_t begin = 0;
                while (begin < text.size() && is_space(text[begin])) ++begin;
                text = text.substr(begin);
                pending_left_strip = false;
            }
            if (!text.empty()) segments.push_back({SegmentKind::Text, text});
        }
        const char close_first = kind == SegmentKind::Expr ? '}' : '%';
        size_t body = next + 2;
        if (body < source.size() && source[body] == '-') {
            ++body;
            strip_previous_tail();
        }
        size_t end = std::string_view::npos;
        for (size_t scan = body; scan + 1 < source.size(); ++scan) {
            if (source[scan] == close_first && source[scan + 1] == '}') {
                end = scan;
                break;
            }
        }
        if (end == std::string_view::npos) return false;
        size_t body_end = end;
        if (body_end > body && source[body_end - 1] == '-') --body_end;
        const std::string_view inner = source.substr(body, body_end - body);
        if (!inner.empty()) segments.push_back({kind, inner});
        pending_left_strip = body_end != end;
        offset = end + 2;
    }
    return true;
}

std::string_view trim(std::string_view text) {
    size_t begin = 0, end = text.size();
    while (begin < end && is_space(text[begin])) ++begin;
    while (end > begin && is_space(text[end - 1])) --end;
    return text.substr(begin, end - begin);
}

// ---------------------------------------------------------------------------
// Expression compilation: a concatenation of string literals and a small set
// of recognized variable references.
// ---------------------------------------------------------------------------

enum class PartKind { Literal, Role, Content, BosToken, EosToken };

struct ExprPart {
    PartKind kind = PartKind::Literal;
    std::string literal;
};

bool parse_expression(std::string_view body, std::vector<ExprPart>& parts) {
    size_t offset = 0;
    bool expect_operand = true;
    while (offset < body.size()) {
        const char c = body[offset];
        if (is_space(c)) {
            ++offset;
            continue;
        }
        if (!expect_operand) {
            if (c != '~' && c != '+') return false;
            ++offset;
            expect_operand = true;
            continue;
        }
        if (c == '\'' || c == '"') {
            const char quote = c;
            size_t end = offset + 1;
            std::string literal;
            while (end < body.size() && body[end] != quote) {
                if (body[end] == '\\' && end + 1 < body.size()) {
                    const char escaped = body[end + 1];
                    switch (escaped) {
                    case 'n': literal.push_back('\n'); break;
                    case 't': literal.push_back('\t'); break;
                    case 'r': literal.push_back('\r'); break;
                    case '\\': literal.push_back('\\'); break;
                    case '\'': literal.push_back('\''); break;
                    case '"': literal.push_back('"'); break;
                    default: return false;
                    }
                    end += 2;
                } else {
                    literal.push_back(body[end]);
                    ++end;
                }
            }
            if (end >= body.size()) return false;
            parts.push_back({PartKind::Literal, std::move(literal)});
            offset = end + 1;
            expect_operand = false;
            continue;
        }
        size_t end = offset;
        while (end < body.size() && !is_space(body[end]) && body[end] != '+' &&
               body[end] != '~')
            ++end;
        const std::string_view name = body.substr(offset, end - offset);
        if (name == "message.role" || name == "message['role']" ||
            name == "message[\"role\"]") {
            parts.push_back({PartKind::Role, {}});
        } else if (name == "message.content" || name == "message['content']" ||
                   name == "message[\"content\"]") {
            parts.push_back({PartKind::Content, {}});
        } else if (name == "bos_token") {
            parts.push_back({PartKind::BosToken, {}});
        } else if (name == "eos_token") {
            parts.push_back({PartKind::EosToken, {}});
        } else {
            return false;
        }
        offset = end;
        expect_operand = false;
    }
    return !parts.empty() && !expect_operand;
}

// A reduced turn expression: literals around content, with an optional
// symbolic role reference between the leading literals.
struct TurnShape {
    std::string prefix;      // literals before the role reference (or content)
    std::string role_mid;    // literals between role reference and content
    std::string suffix;      // literals after content
    bool role_symbolic = false;
};

bool turn_shape_from_parts(const std::vector<ExprPart>& parts, TurnShape& shape) {
    bool saw_content = false;
    for (const ExprPart& part : parts) {
        switch (part.kind) {
        case PartKind::Literal:
            if (!shape.role_symbolic && !saw_content) {
                shape.prefix += part.literal;
            } else if (shape.role_symbolic && !saw_content) {
                shape.role_mid += part.literal;
            } else {
                shape.suffix += part.literal;
            }
            break;
        case PartKind::Role:
            if (shape.role_symbolic || saw_content) return false;
            shape.role_symbolic = true;
            break;
        case PartKind::Content:
            if (saw_content) return false;
            saw_content = true;
            break;
        case PartKind::BosToken:
        case PartKind::EosToken:
            return false;
        }
    }
    return saw_content;
}

enum class LiteralAllow { LiteralsOnly, AllowBos, AllowEos };

bool literal_from_parts(const std::vector<ExprPart>& parts, LiteralAllow allow,
                        std::string& out, bool& saw_bos, bool& saw_eos) {
    for (const ExprPart& part : parts) {
        switch (part.kind) {
        case PartKind::Literal:
            out += part.literal;
            break;
        case PartKind::BosToken:
            if (allow != LiteralAllow::AllowBos) return false;
            saw_bos = true;
            break;
        case PartKind::EosToken:
            if (allow != LiteralAllow::AllowEos) return false;
            saw_eos = true;
            break;
        default:
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Structural walker.
// ---------------------------------------------------------------------------

struct Walker {
    const std::vector<Segment>& segments;
    size_t index = 0;

    bool done() const { return index >= segments.size(); }
    const Segment* peek() const { return done() ? nullptr : &segments[index]; }
    bool at_tag(std::string_view keyword) const {
        const Segment* segment = peek();
        return segment && segment->kind == SegmentKind::Tag &&
               trim(segment->text) == keyword;
    }
    bool at_tag_starting(std::string_view prefix) const {
        const Segment* segment = peek();
        return segment && segment->kind == SegmentKind::Tag &&
               trim(segment->text).substr(0, prefix.size()) == prefix;
    }
    bool accept_tag(std::string_view keyword) {
        if (!at_tag(keyword)) return false;
        ++index;
        return true;
    }
    std::string_view peek_tag() const {
        const Segment* segment = peek();
        return segment && segment->kind == SegmentKind::Tag ? trim(segment->text)
                                                           : std::string_view{};
    }
    // Collects literal text from consecutive text/expr segments until the
    // next tag. Non-literal expressions fail closed per `allow`.
    bool collect_literals_until_tag(std::string& out, LiteralAllow allow, bool& saw_bos,
                                    bool& saw_eos) {
        while (!done() && segments[index].kind != SegmentKind::Tag) {
            if (segments[index].kind == SegmentKind::Text) {
                out += segments[index].text;
                ++index;
                continue;
            }
            std::vector<ExprPart> parts;
            if (!parse_expression(segments[index].text, parts) ||
                !literal_from_parts(parts, allow, out, saw_bos, saw_eos))
                return false;
            ++index;
        }
        return true;
    }
};

bool mentions(std::string_view text, std::string_view token) {
    return text.find(token) != std::string_view::npos;
}

// Extracts the string literal compared against in `... == 'name' ...`.
bool role_literal_in_guard(std::string_view guard, std::string& role) {
    const size_t eq = guard.find("==");
    if (eq == std::string_view::npos) return false;
    const std::string_view rest = trim(guard.substr(eq + 2));
    if (rest.empty() || (rest.front() != '\'' && rest.front() != '"')) return false;
    const char quote = rest.front();
    const size_t end = rest.find(quote, 1);
    if (end == std::string_view::npos) return false;
    role = std::string(rest.substr(1, end - 1));
    return !role.empty();
}

struct FramingState {
    ChatFraming framing;
    bool have_turn = false;
};

// Parses one branch body (up to the next elif/else/endif) as a turn
// expression. `role_symbolic` selects between ChatML-style shared patterns
// (the role reference stays symbolic) and role-keyed literal patterns.
bool parse_turn_branch(Walker& walker, TurnShape& shape) {
    if (walker.done() || walker.peek()->kind != SegmentKind::Expr) return false;
    std::vector<ExprPart> parts;
    if (!parse_expression(walker.peek()->text, parts)) return false;
    if (!turn_shape_from_parts(parts, shape)) return false;
    ++walker.index;
    std::string trailing;
    bool saw_bos = false, saw_eos = false;
    if (!walker.collect_literals_until_tag(trailing, LiteralAllow::LiteralsOnly, saw_bos,
                                          saw_eos))
        return false;
    shape.suffix += trailing;
    return true;
}

bool process_loop_body(const std::vector<Segment>& body, FramingState& state) {
    Walker walker{body, 0};
    ChatFraming& framing = state.framing;
    std::string user_open, assistant_open, turn_close;
    bool have_user = false, have_assistant = false;

    // Skips a branch body up to the next elif/else/endif at the same depth,
    // tracking nested if/endif and for/endfor pairs.
    const auto skip_branch_body = [&walker]() {
        int depth = 0;
        while (!walker.done()) {
            if (walker.peek()->kind == SegmentKind::Tag) {
                const std::string_view tag = walker.peek_tag();
                const bool opener = tag.substr(0, 3) == "if " || tag.substr(0, 4) == "for ";
                const bool closer = tag == "endif" || tag == "endfor";
                if (depth == 0 &&
                    (tag.substr(0, 5) == "elif " || tag == "else" || closer))
                    return true;
                if (opener) ++depth;
                else if (closer) --depth;
            }
            ++walker.index;
        }
        return false;
    };

    // Optional in-loop default-system block:
    //   {% if loop.first and messages[0]['role'] != 'system' %} DEFAULT {% endif %}
    if (walker.at_tag_starting("if ")) {
        const std::string_view guard = walker.peek_tag().substr(3);
        if (mentions(guard, "loop.first") && mentions(guard, "system") &&
            mentions(guard, "!=")) {
            ++walker.index;
            std::string default_system;
            bool saw_bos = false, saw_eos = false;
            if (!walker.collect_literals_until_tag(default_system, LiteralAllow::LiteralsOnly,
                                                   saw_bos, saw_eos) ||
                !walker.accept_tag("endif"))
                return false;
            framing.system_prefix = std::move(default_system);
        }
    }

    if (walker.done()) return false;

    if (walker.peek()->kind == SegmentKind::Expr) {
        // Form (a): one unconditional expression, role symbolic.
        TurnShape shape;
        if (!parse_turn_branch(walker, shape) || !shape.role_symbolic) return false;
        user_open = shape.prefix + "user" + shape.role_mid;
        assistant_open = shape.prefix + "assistant" + shape.role_mid;
        turn_close = shape.suffix;
        have_user = have_assistant = true;
    } else if (walker.at_tag_starting("if ")) {
        const std::string_view guard = walker.peek_tag().substr(3);
        // A complex primary guard offers alternative roles ("user" or plain
        // "system" or plain "assistant") and shares one ChatML-style pattern;
        // it must be detected before the role-keyed form because it also
        // contains == comparisons.
        const bool primary_complex =
            mentions(guard, "message.role") && mentions(guard, " or ");
        std::string first_role;
        const bool keyed_on_role =
            !primary_complex && role_literal_in_guard(guard, first_role);
        if (!keyed_on_role && !primary_complex) return false;
        ++walker.index;

        // Parses the current branch as a turn expression and records it.
        // An empty role marks the primary shared pattern, whose symbolic
        // role reference yields both markers at once.
        const auto record_branch = [&](const std::string& role, bool require_symbolic) {
            TurnShape shape;
            if (!parse_turn_branch(walker, shape)) return false;
            if (shape.role_symbolic != require_symbolic) return false;
            if (role.empty()) {
                user_open = shape.prefix + "user" + shape.role_mid;
                assistant_open = shape.prefix + "assistant" + shape.role_mid;
                turn_close = shape.suffix;
                have_user = have_assistant = true;
            } else if (role == "user") {
                user_open = shape.prefix + shape.role_mid;
                turn_close = shape.suffix;
                have_user = true;
            } else if (role == "assistant") {
                assistant_open = shape.prefix + shape.role_mid;
                turn_close = shape.suffix;
                have_assistant = true;
            }
            return true;
        };

        if (primary_complex) {
            if (!record_branch({}, true)) return false;
        } else if (first_role == "user" || first_role == "assistant") {
            if (!record_branch(first_role, false)) return false;
        } else if (first_role == "system") {
            // A system branch frames explicit system turns; plain
            // conversations carry no system message, so it is skipped.
            if (!skip_branch_body()) return false;
        }

        // Remaining elif/else branches. A role-keyed user/assistant branch
        // is parsed as a turn branch unless the primary guard already
        // covered that role: by if/elif semantics a covered-role elif only
        // fires for messages the primary excluded (tool conversations), so
        // it is a refinement and is skipped depth-aware. Tool-scoped guards
        // are skipped the same way; anything else fails closed.
        while (walker.at_tag_starting("elif ") || walker.at_tag_starting("else")) {
            const std::string_view branch = walker.peek_tag();
            ++walker.index;
            if (branch.substr(0, 5) == "elif ") {
                const std::string_view guard = branch.substr(5);
                std::string elif_role;
                const bool role_keyed = role_literal_in_guard(guard, elif_role);
                const bool uncovered_turn_role =
                    role_keyed &&
                    ((elif_role == "user" && !have_user) ||
                     (elif_role == "assistant" && !have_assistant));
                if (uncovered_turn_role) {
                    if (!record_branch(elif_role, false)) return false;
                    continue;
                }
                const bool covered_refinement =
                    role_keyed &&
                    ((elif_role == "user" && have_user) ||
                     (elif_role == "assistant" && have_assistant) ||
                     elif_role == "system");
                if (!covered_refinement && !mentions(guard, "tool")) return false;
            }
            if (!skip_branch_body()) return false;
            if (walker.at_tag("endif")) break;
        }
        if (!walker.accept_tag("endif")) return false;
    } else {
        return false;
    }

    // Only trailing literal text may remain in the loop body.
    std::string tail;
    bool saw_bos = false, saw_eos = false;
    if (!walker.collect_literals_until_tag(tail, LiteralAllow::LiteralsOnly, saw_bos, saw_eos))
        return false;
    if (!tail.empty()) turn_close += tail;
    if (!walker.done()) return false;

    if (have_user && have_assistant && !turn_close.empty()) {
        framing.user_open = std::move(user_open);
        framing.assistant_open = std::move(assistant_open);
        framing.turn_close = std::move(turn_close);
        state.have_turn = true;
    }
    return state.have_turn;
}

}  // namespace

ChatFraming compile_chat_framing(std::string_view template_text) {
    ChatFraming framing;
    if (template_text.empty() || template_text.size() > 256u * 1024u) return framing;
    std::vector<Segment> segments;
    if (!lex_template(template_text, segments)) return framing;

    FramingState state;
    Walker walker{segments, 0};

    // ---- Prologue: everything before the message loop. ----
    bool bos_emitted = false;
    // Depth of outer `if tools` blocks whose else branch the prologue is
    // currently inside; their matching endif tags must be consumed.
    int pending_outer_endif = 0;
    while (!walker.done()) {
        if (walker.at_tag_starting("for ")) break;
        if (walker.at_tag("endif") && pending_outer_endif > 0) {
            --pending_outer_endif;
            ++walker.index;
            continue;
        }
        if (walker.at_tag_starting("if ")) {
            const std::string_view guard = walker.peek_tag().substr(3);
            // Outer tool preambles: plain chat takes the else path; a
            // tools-only template without an else fails closed below.
            if (guard == "tools" || mentions(guard, "tool_calls")) {
                ++walker.index;
                int depth = 1;
                bool took_else = false;
                while (!walker.done() && depth > 0) {
                    if (walker.peek()->kind != SegmentKind::Tag) {
                        ++walker.index;
                        continue;
                    }
                    const std::string_view tag = walker.peek_tag();
                    if (tag.substr(0, 3) == "if ") {
                        ++depth;
                        ++walker.index;
                        continue;
                    }
                    if (tag == "endif") {
                        --depth;
                        ++walker.index;
                        continue;
                    }
                    if (depth == 1 && tag == "else") {
                        took_else = true;
                        ++walker.index;
                        // The else branch becomes the new prologue: restart
                        // the prologue scan from here at the same depth.
                        break;
                    }
                    ++walker.index;
                }
                if (!took_else || walker.done()) return framing;
                ++pending_outer_endif;
                continue;
            }
            // System selection: messages[0] system content or a default.
            if (mentions(guard, "messages[0]") && mentions(guard, "system")) {
                ++walker.index;
                // The explicit-system branch references messages[0] content
                // and is skipped; plain conversations use the default.
                int depth = 0;
                while (!walker.done()) {
                    if (walker.peek()->kind == SegmentKind::Tag) {
                        const std::string_view tag = walker.peek_tag();
                        if (depth == 0 && (tag == "else" || tag == "endif")) break;
                        if (tag.substr(0, 3) == "if ") ++depth;
                        if (depth > 0 && tag == "endif") --depth;
                    }
                    ++walker.index;
                }
                if (walker.done()) return framing;
                std::string default_system;
                bool saw_bos = false, saw_eos = false;
                if (walker.accept_tag("else")) {
                    if (!walker.collect_literals_until_tag(default_system,
                                                           LiteralAllow::LiteralsOnly,
                                                           saw_bos, saw_eos) ||
                        !walker.accept_tag("endif"))
                        return framing;
                    if (default_system.empty() || !state.framing.system_prefix.empty())
                        return framing;
                    state.framing.system_prefix = std::move(default_system);
                } else if (!walker.accept_tag("endif")) {
                    return framing;
                }
                continue;
            }
            return framing;  // unrecognized prologue conditional: fail closed
        }
        if (walker.peek()->kind == SegmentKind::Expr) {
            std::vector<ExprPart> parts;
            if (!parse_expression(walker.peek()->text, parts)) return framing;
            bool saw_bos = false, saw_eos = false;
            std::string literal;
            if (!literal_from_parts(parts, LiteralAllow::AllowBos, literal, saw_bos, saw_eos))
                return framing;
            if (saw_eos || (!literal.empty() && !saw_bos)) return framing;
            if (saw_bos) bos_emitted = true;
            ++walker.index;
            continue;
        }
        if (walker.peek()->kind == SegmentKind::Text) {
            if (!trim(walker.peek()->text).empty()) return framing;
            ++walker.index;
            continue;
        }
        return framing;
    }

    // ---- Main message loop. ----
    if (!walker.at_tag_starting("for ")) return framing;
    if (!mentions(walker.peek_tag(), "message in messages")) return framing;
    ++walker.index;
    std::vector<Segment> body;
    int depth = 1;
    while (!walker.done()) {
        if (walker.peek()->kind == SegmentKind::Tag) {
            const std::string_view tag = walker.peek_tag();
            if (tag.substr(0, 4) == "for ") ++depth;
            if (tag == "endfor") {
                --depth;
                if (depth == 0) {
                    ++walker.index;
                    break;
                }
            }
        }
        body.push_back(segments[walker.index]);
        ++walker.index;
    }
    if (depth != 0 || !process_loop_body(body, state)) return framing;

    // ---- Generation prompt block. ----
    if (!walker.at_tag_starting("if ")) return framing;
    if (!mentions(walker.peek_tag(), "add_generation_prompt")) return framing;
    ++walker.index;
    std::string generation_open;
    bool saw_bos = false, saw_eos = false;
    if (!walker.collect_literals_until_tag(generation_open, LiteralAllow::LiteralsOnly,
                                           saw_bos, saw_eos))
        return framing;
    // An optional else branch (typically the eos token) is not needed for
    // generation framing.
    if (walker.at_tag_starting("else")) {
        ++walker.index;
        while (!walker.done() && !walker.at_tag("endif")) {
            if (walker.peek()->kind == SegmentKind::Tag) return framing;
            ++walker.index;
        }
    }
    if (!walker.accept_tag("endif")) return framing;

    while (!walker.done()) {
        if (walker.peek()->kind != SegmentKind::Text || !trim(walker.peek()->text).empty())
            return framing;
        ++walker.index;
    }

    if (generation_open.empty()) return framing;
    state.framing.generation_open = std::move(generation_open);
    state.framing.template_emits_bos = bos_emitted;
    state.framing.matched = true;
    return state.framing;
}

}  // namespace Laplace
