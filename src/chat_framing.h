// chat_framing.h - structural chat-template compilation.
//
// GGUF packages carry their instruct framing as a Jinja chat template in
// tokenizer.chat_template. Rather than interpreting Jinja at runtime, the
// common conversational shapes are compiled once, at package load, into
// declared framing facts. Every fact comes from the template's own structure;
// nothing is keyed on a model or family name, and any template outside the
// recognized shapes fails closed to the neutral newline marker.
#pragma once

#include <string>
#include <string_view>

namespace Laplace {

struct ChatFraming {
    bool matched = false;
    // The template itself emits {{ bos_token }} in its prologue. The BOS
    // token itself comes from the package's own tokenizer facts: when the
    // package already declares an automatic BOS prefix, the framing defers
    // to it and drops the emission; otherwise the framing owns it.
    bool template_emits_bos = false;
    // Default system block exactly as the template would render it for a
    // conversation with no system message (markers included). Empty when
    // the template has no default system content.
    std::string system_prefix;
    // Turn markers. user_open/assistant_open include the role display name
    // exactly as the template writes it (e.g. "model" for gemma assistants).
    std::string user_open;
    std::string assistant_open;
    std::string turn_close;
    // Opening marker for the assistant's generated turn.
    std::string generation_open;

    friend bool operator==(const ChatFraming&, const ChatFraming&) = default;
};

// Compiles the framing facts from a chat template. Returns
// ChatFraming{.matched = false} for anything outside the recognized
// conversational shapes; never throws.
ChatFraming compile_chat_framing(std::string_view template_text);

}  // namespace Laplace
