// Exact reviewed LAPRUL10 bytes. No compression or runtime reconstruction.
#include "compat_rules_v1_data.inc"

std::span<const uint8_t> bundled_compatibility_rule_record() {
    return kBundledCompatibilityRuleV1Bytes;
}

const std::vector<CompatibilityRule>& bundled_compatibility_rules() {
    static const std::vector<CompatibilityRule> rules = [] {
        const std::span<const uint8_t> record = bundled_compatibility_rule_record();
        std::vector<uint8_t> bytes(record.begin(), record.end());
        auto decoded = decode_compatibility_rule(bytes);
        if (!std::holds_alternative<CompatibilityRule>(decoded)) return std::vector<CompatibilityRule>{};
        return std::vector<CompatibilityRule>{std::get<CompatibilityRule>(std::move(decoded))};
    }();
    return rules;
}
