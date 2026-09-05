#include "semantic_program_compiler.h"
#include "program_reference.h"
#include "test_util.h"

#include <bit>
#include <cmath>
#include <limits>
#include <tuple>
#include <unordered_map>

#ifdef LAPLACE_SEMANTIC_PROGRAM_METAL
#include "program_metal.h"
#endif

using namespace Laplace;
namespace {
using Data = std::unordered_map<uint32_t, ReferenceValue>;
uint32_t bits(float f) { return std::bit_cast<uint32_t>(f); }
std::vector<Dimension> shape(std::initializer_list<uint64_t> dims) {
    std::vector<Dimension> result;
    for (auto d : dims) result.push_back({DimensionKind::Constant, d});
    return result;
}
SemanticValue value(uint32_t id, std::initializer_list<uint64_t> dims, ScalarType t=ScalarType::F32) {
    return {id,t,shape(dims),0};
}
SemanticTensor tensor(uint32_t id, std::initializer_list<uint64_t> dims) {
    SemanticTensor t; t.id=id; t.dimensions=shape(dims); return t;
}
ReferenceValue data(std::initializer_list<uint64_t> dims, const std::vector<float>& v) {
    ReferenceValue r{ElementType::F32,dims,{}};
    for (float f:v) r.bits.push_back(bits(f));
    return r;
}
SemanticOperator op(uint32_t id,OperatorKind kind,std::vector<uint32_t> in,std::vector<uint32_t> out,
                    std::vector<uint32_t> tensors,OperatorPayload payload) {
    return {id,kind,8,std::move(in),std::move(out),std::move(tensors),{},std::move(payload)};
}
SemanticModel model(std::vector<SemanticValue> values,uint32_t inputs,std::vector<SemanticOperator> operators,
                    std::vector<SemanticTensor> tensors={}) {
    SemanticModel m; m.values=std::move(values); m.input_values_count=inputs;
    m.output_values_first=m.values.size()-1; m.output_values_count=1;
    m.operators=std::move(operators);m.tensors=std::move(tensors);return m;
}
std::vector<float> floats(const SemanticVectorResult& v) {
    CHECK(std::holds_alternative<std::vector<float>>(v));
    return std::holds_alternative<std::vector<float>>(v)?std::get<std::vector<float>>(v):std::vector<float>{};
}
ReferenceExecutionResult run(const SemanticModel& m,const Data& inputs,const Data& weights,
                             std::span<const SemanticDimensionBinding> dimensions={}) {
    auto compiled=compile_semantic_program(m,dimensions);
    const auto* error=std::get_if<CompatibilityReport>(&compiled);
    CHECK_MSG(!error,"compile: %s",error?error->detail.c_str():"");
    if(error)return *error;
    auto& c=std::get<CompiledSemanticProgram>(compiled);
    CHECK(c.outputs==std::vector<uint32_t>{m.values[m.output_values_first].id});
    std::vector<ReferenceInput> args;
    for(auto b:c.inputs)args.push_back({b.entry_value_id,inputs.at(b.semantic_id)});
    for(auto b:c.tensors)args.push_back({b.entry_value_id,weights.at(b.semantic_id)});
    ReferenceState state;
    auto result=execute_reference_program(c.program,state,args);
#ifdef LAPLACE_SEMANTIC_PROGRAM_METAL
    auto native=compile_metal_program(c.program);
    const auto* native_error=std::get_if<CompatibilityReport>(&native);
    CHECK_MSG(!native_error,"native compile: %s",native_error?native_error->detail.c_str():"");
    if(!native_error) {
        std::vector<MetalProgramInput> native_inputs;
        for(const auto& a:args)native_inputs.push_back({a.value_id,{a.value.type,a.value.extents,a.value.bits}});
        auto execution=std::get<MetalProgramExecutable>(native).execute(native_inputs);
        CHECK(std::holds_alternative<MetalProgramResult>(execution)==std::holds_alternative<ReferenceResult>(result));
        if(const auto* output=std::get_if<MetalProgramResult>(&execution)) {
            const auto& ref=std::get<ReferenceResult>(result).exports.front();
            CHECK(output->exports.front().extents==ref.extents);
            CHECK(output->exports.front().bits.size()==ref.bits.size());
            for(size_t i=0;i<ref.bits.size();++i) {
                auto actual=std::bit_cast<float>(uint32_t(output->exports.front().bits[i]));
                auto expected=std::bit_cast<float>(uint32_t(ref.bits[i]));
                // Native transcendental instructions need not have host libm's last bit.
                CHECK(actual==expected || (std::isnan(actual)&&std::isnan(expected)) ||
                      std::abs(actual-expected)<=2e-6f*std::max(1.0f,std::abs(expected)));
            }
        }
    }
#endif
    return result;
}
void equal(const ReferenceExecutionResult& result,const std::vector<float>& expected) {
    const auto* r=std::get_if<ReferenceResult>(&result);
    const auto* error=std::get_if<CompatibilityReport>(&result);
    CHECK_MSG(r,"execute: %s",error?error->detail.c_str():"");
    if(!r)return;
    CHECK(r->exports.size()==1);CHECK(r->exports.front().bits.size()==expected.size());
    for(size_t i=0;i<expected.size()&&i<r->exports.front().bits.size();++i)
        CHECK_MSG(r->exports.front().bits[i]==bits(expected[i]),"element %zu actual=%08llx expected=%08x",i,
            static_cast<unsigned long long>(r->exports.front().bits[i]),bits(expected[i]));
}
void test_linear_association() {
    for(uint64_t k:{1,15,16,17,31,32,33})for(bool transpose:{false,true}) {
        auto m=model({value(91,{2,k}),value(92,{2,3})},1,
            {op(100,OperatorKind::Linear,{91},{92},{701,702},LinearPayload{transpose,true,ScalarType::F32})},
            {transpose?tensor(701,{3,k}):tensor(701,{k,3}),tensor(702,{3})});
        std::vector<float> x(2*k),w(3*k),stored(3*k),bias{0.25f,-0.5f,0.125f};
        for(size_t i=0;i<x.size();++i)x[i]=float(int(i%11)-5)*0.125f;
        for(size_t o=0;o<3;++o)for(size_t i=0;i<k;++i) {
            w[o*k+i]=float(int((i*7+o)%13)-6)*0.0625f;
            stored[transpose?o*k+i:i*3+o]=w[o*k+i];
        }
        std::vector<float> expected;
        for(size_t row=0;row<2;++row) {
            auto v=floats(semantic_linear({x.begin()+row*k,x.begin()+(row+1)*k},w,3,k,bias));
            expected.insert(expected.end(),v.begin(),v.end());
        }
        auto wd=transpose?data({3,k},stored):data({k,3},stored);
        equal(run(m,{{91,data({2,k},x)}},{{701,wd},{702,data({3},bias)}}),expected);
    }
}
void test_norms() {
    for(uint64_t width:{1,15,16,17,32,33})for(auto kind:{OperatorKind::RmsNorm,OperatorKind::L2Normalize,OperatorKind::GatedRmsNorm}) {
        std::vector<float>x(width),w(width),gate(width);
        for(size_t i=0;i<width;++i){x[i]=float(int(i%7)-3)*0.3f;w[i]=0.8f+float(i%3)*0.1f;gate[i]=float(int(i%5)-2);}
        float epsilon=0.0001f;
        auto m=model({value(8,{width}),value(9,{width}),value(10,{width})},2,{},{});
        if(kind==OperatorKind::RmsNorm){m.operators={op(0,kind,{8},{10},{4},RmsNormPayload{bits(epsilon)})};m.tensors={tensor(4,{width})};}
        else if(kind==OperatorKind::GatedRmsNorm){m.operators={op(0,kind,{8,9},{10},{4},GatedRmsNormPayload{bits(epsilon)})};m.tensors={tensor(4,{width})};}
        else m.operators={op(0,kind,{8},{10},{},L2NormalizePayload{bits(epsilon)})};
        auto expected=kind==OperatorKind::RmsNorm?semantic_rms_norm(x,w,epsilon):kind==OperatorKind::GatedRmsNorm?semantic_gated_rms_norm(x,w,gate,epsilon):semantic_l2_normalize(x,epsilon);
        equal(run(m,{{8,data({width},x)},{9,data({width},gate)}},{{4,data({width},w)}}),floats(expected));
    }
}
void test_activation_and_composition() {
    std::vector<float>x{-100,-3,-0.0f,0,1,4},y{1,2,3,4,5,6};
    for(auto kind:{OperatorKind::SwiGlu,OperatorKind::GatedActivation,OperatorKind::GatedAttention})for(auto activation:{ActivationKind::Silu,ActivationKind::GeluTanh}) {
        if(kind!=OperatorKind::GatedActivation&&activation==ActivationKind::GeluTanh)continue;
        OperatorPayload p=kind==OperatorKind::SwiGlu?OperatorPayload{SwiGluPayload{activation}}:kind==OperatorKind::GatedActivation?OperatorPayload{GatedActivationPayload{activation}}:OperatorPayload{GatedAttentionPayload{}};
        auto m=model({value(0,{6}),value(1,{6}),value(2,{6})},2,{op(5,kind,{0,1},{2},{},p)});
        auto expected=kind==OperatorKind::GatedAttention?semantic_gated_attention_output(x,y):semantic_gated_activation(x,y,activation);
        equal(run(m,{{0,data({6},x)},{1,data({6},y)}},{}),floats(expected));
    }
    auto m=model({value(10,{2,4}),value(11,{2,2}),value(12,{2,2}),value(13,{2,4}),value(14,{2,4}),value(15,{2,4}),value(16,{2,4})},1,
        {op(0,OperatorKind::AxisSplit,{10},{11,12},{},AxisSplitPayload{2,2}),
         op(1,OperatorKind::Concat,{12,11},{13},{},ConcatPayload{}),
         op(2,OperatorKind::Scale,{13},{14},{},ScalePayload{ScaleSource::LiteralF32,bits(0.5f)}),
         op(3,OperatorKind::Add,{14,13},{15},{},AddPayload{}),
         op(4,OperatorKind::TanhSoftcap,{15},{16},{},TanhSoftcapPayload{bits(3)})});
    std::vector<float>v{1,-0.0f,-2,-0.0f,3,4,5,6};
    auto split=std::get<SemanticAxisSplit>(semantic_axis_split(v,4,2));
    auto concat=floats(semantic_concat_last_axis(split.second,split.first,2,2));
    auto scaled=floats(semantic_scale(concat,0.5));
    equal(run(m,{{10,data({2,4},v)}},{}),floats(semantic_tanh_softcap(floats(semantic_add(scaled,concat)),3)));
    m.operators.resize(2);m.output_values_first=3;
    equal(run(m,{{10,data({2,4},v)}},{}),concat);
}
void test_embedding_and_binding() {
    auto m=model({value(123,{2},ScalarType::U32),value(456,{2,3}),value(789,{2,3})},1,
        {op(9,OperatorKind::EmbeddingLookup,{123},{456},{88},EmbeddingLookupPayload{bits(0.5),4,3}),
         op(10,OperatorKind::Scale,{456},{789},{99},ScalePayload{ScaleSource::Tensor,0})},
         {tensor(88,{3,4}),tensor(99,{1})});
    std::vector<float>table{1,4,7,10,2,5,8,11,3,6,9,12};
    equal(run(m,{{123,{ElementType::U32,{2},{2,0}}}},{{88,data({3,4},table)},{99,data({1},{2})}}),{7,8,9,1,2,3});
    auto bad=run(m,{{123,{ElementType::U32,{2},{4,0}}}},{{88,data({3,4},table)},{99,data({1},{2})}});
    CHECK(std::holds_alternative<CompatibilityReport>(bad));
    m.values[0].dimensions[0]={DimensionKind::Symbol,77};m.values[1].dimensions[0]={DimensionKind::Symbol,77};m.values[2].dimensions[0]={DimensionKind::Symbol,77};
    CHECK(std::holds_alternative<CompatibilityReport>(compile_semantic_program(m)));
    SemanticDimensionBinding binding{77,2};
    equal(run(m,{{123,{ElementType::U32,{2},{2,0}}}},{{88,data({3,4},table)},{99,data({1},{2})}},std::span(&binding,1)),{7,8,9,1,2,3});
}
void test_grouped_norm_and_shape_rejection() {
    auto m=model({value(21,{2,34}),value(22,{2,34})},1,
        {op(7,OperatorKind::RmsNorm,{21},{22},{6},RmsNormPayload{bits(0.0001f),-1,1,RmsNormAffineGeometry::SharedAcrossGroups,17})},
        {tensor(6,{17})});
    std::vector<float>x(68),w(17),expected;
    for(size_t i=0;i<x.size();++i)x[i]=float(int(i%13)-6)*0.2f;
    for(size_t i=0;i<w.size();++i)w[i]=0.5f+float(i%5)*0.3f;
    for(size_t i=0;i<4;++i){auto y=floats(semantic_rms_norm({x.begin()+17*i,x.begin()+17*(i+1)},w,0.0001f));expected.insert(expected.end(),y.begin(),y.end());}
    equal(run(m,{{21,data({2,34},x)}},{{6,data({17},w)}}),expected);
    m.tensors[0].dimensions=shape({16});CHECK(std::holds_alternative<CompatibilityReport>(compile_semantic_program(m)));
    m.tensors[0].dimensions=shape({17});m.values[1].id=21;CHECK(std::holds_alternative<CompatibilityReport>(compile_semantic_program(m)));
}
void test_stateful_rope_attention() {
    std::vector<std::tuple<RopePairing,uint32_t,AttentionWindowKind,ValueSource>> cases;
    for(auto pairing:{RopePairing::HalfSplit,RopePairing::Interleaved})for(uint32_t dh:{4u,17u})
        cases.emplace_back(pairing,dh,AttentionWindowKind::Global,ValueSource::SeparateProjection);
    for(auto window:{AttentionWindowKind::Global,AttentionWindowKind::Sliding})
        for(auto source:{ValueSource::SeparateProjection,ValueSource::KeyPreRope,ValueSource::KeyPostRope,ValueSource::KeyStateAlias})
            if(window!=AttentionWindowKind::Global || source!=ValueSource::SeparateProjection)
                cases.emplace_back(RopePairing::HalfSplit,4,window,source);
    for(auto [pairing,dh,window,source]:cases) {
        const uint32_t qh=4,kh=2,capacity=6,rotary=4;
        auto m=model({value(0,{1,qh*dh}),value(1,{1,kh*dh}),value(2,{1,kh*dh}),
            value(3,{1,qh*dh}),value(4,{1,kh*dh}),value(5,{1,qh*dh}),value(6,{1,qh*dh})},3,
            {op(0,OperatorKind::Rope,{0,1},{3,4},{},RopePayload{pairing,true,rotary,bits(10000),bits(0.75f),{},8}),
             op(1,OperatorKind::CausalAttention,{3,4,2},{5},{},CausalAttentionPayload{qh,kh,dh,bits(0.5f)}),
             op(2,OperatorKind::Scale,{5},{6},{100},ScalePayload{ScaleSource::Tensor,0})}, {tensor(100,{1})});
        m.maximum_context=capacity;
        m.states={{91,StateKind::KeyCache,1,StateUpdateKind::AppendKey,PositionPolicy::AppendOnly,shape({capacity,kh,dh}),{},0},
                  {92,StateKind::ValueCache,1,StateUpdateKind::AppendValue,PositionPolicy::AppendOnly,shape({capacity,kh,dh}),{},0}};
        m.operators[1].states={91,92};
        auto& attention=std::get<CausalAttentionPayload>(m.operators[1].payload);
        attention.window=window;attention.window_tokens=window==AttentionWindowKind::Sliding?2:0;
        attention.value_source=source;
        attention.value_source_value=source==ValueSource::SeparateProjection?2:source==ValueSource::KeyPreRope?1:4;
        if(source!=ValueSource::SeparateProjection)m.operators[1].inputs={3,4};
        if(source==ValueSource::KeyStateAlias){m.operators[1].states={91};m.states.resize(1);}
        auto malformed_key_output=m;
        malformed_key_output.values[4].dimensions.back()={DimensionKind::Constant,qh*dh};
        CHECK(std::holds_alternative<CompatibilityReport>(compile_semantic_program(malformed_key_output)));
        auto compiled=compile_semantic_program(m);
        const auto* compile_error=std::get_if<CompatibilityReport>(&compiled);
        CHECK_MSG(!compile_error,"stateful compile: %s",compile_error?compile_error->detail.c_str():"");
        if(compile_error)continue;
        const auto& c=std::get<CompiledSemanticProgram>(compiled);
        CHECK(program_definition(c.program).state_references.size()==(source==ValueSource::KeyStateAlias?2:3));
        ReferenceState state;SemanticKvState kv;
#ifdef LAPLACE_SEMANTIC_PROGRAM_METAL
        auto native=compile_metal_program(c.program);
        const auto* native_error=std::get_if<CompatibilityReport>(&native);
        CHECK_MSG(!native_error,"stateful native compile: %s",native_error?native_error->detail.c_str():"");
#endif
        for(uint32_t position=0;position<=capacity;++position) {
            std::vector<float>q(qh*dh),k(kh*dh),v(kh*dh);
            for(size_t i=0;i<q.size();++i)q[i]=float(int((i*7+position*3)%19)-9)*0.08f;
            for(size_t i=0;i<k.size();++i){k[i]=float(int((i*3+position*5)%17)-8)*0.07f;v[i]=float(int((i*11+position)%13)-6)*0.2f;}
            const auto rotate=[&](std::vector<float> x) {
                for(size_t h=0;h<x.size()/dh;++h) {
                    std::vector<float>head(x.begin()+h*dh,x.begin()+(h+1)*dh);
                    auto result=pairing==RopePairing::HalfSplit?semantic_rope_half_split(head,head,position,rotary,10000,0.75f,8):semantic_rope_interleaved(head,head,position,rotary,10000,0.75f,8);
                    auto rotated=floats(result);std::copy(rotated.begin(),rotated.begin()+dh,x.begin()+h*dh);
                }
                return x;
            };
            std::vector<ReferenceInput> args;
            const Data inputs{{0,data({1,qh*dh},q)},{1,data({1,kh*dh},k)},{2,data({1,kh*dh},v)}};
            for(auto b:c.inputs)args.push_back({b.entry_value_id,inputs.at(b.semantic_id)});
            for(auto b:c.tensors)args.push_back({b.entry_value_id,data({1},{1})});
            auto before=state;
            if(position==2) {
                auto invalid=args;invalid.back().value.bits[0]=bits(std::numeric_limits<float>::quiet_NaN());
                auto rejected=execute_reference_program(c.program,state,invalid);
                CHECK(std::holds_alternative<CompatibilityReport>(rejected));CHECK(state.slots==before.slots);CHECK(state.generation==before.generation);
#ifdef LAPLACE_SEMANTIC_PROGRAM_METAL
                if(!native_error) {
                    std::vector<MetalProgramInput> native_invalid;
                    for(const auto& a:invalid)native_invalid.push_back({a.value_id,{a.value.type,a.value.extents,a.value.bits}});
                    CHECK(std::holds_alternative<CompatibilityReport>(std::get<MetalProgramExecutable>(native).execute(native_invalid)));
                }
#endif
            }
            auto result=execute_reference_program(c.program,state,args);
            if(position==capacity) {CHECK(std::holds_alternative<CompatibilityReport>(result));CHECK(state.slots==before.slots);CHECK(state.generation==before.generation);continue;}
            auto value_input=source==ValueSource::SeparateProjection?v:source==ValueSource::KeyPreRope?k:rotate(k);
            auto expected=floats(semantic_causal_attention_windowed(rotate(q),rotate(k),value_input,1,qh,kh,dh,0.5f,window,attention.window_tokens,kv));
            equal(result,expected);
#ifdef LAPLACE_SEMANTIC_PROGRAM_METAL
            if(!native_error) {
                std::vector<MetalProgramInput> native_inputs;
                for(const auto& a:args)native_inputs.push_back({a.value_id,{a.value.type,a.value.extents,a.value.bits}});
                auto executed=std::get<MetalProgramExecutable>(native).execute(native_inputs);
                const auto* output=std::get_if<MetalProgramResult>(&executed);
                const auto* e=std::get_if<CompatibilityReport>(&executed);
                CHECK_MSG(output,"stateful native execute: %s",e?e->detail.c_str():"");
                if(output)for(size_t i=0;i<expected.size();++i) {
                    float actual=std::bit_cast<float>(uint32_t(output->exports.front().bits[i]));
                    CHECK(std::abs(actual-expected[i])<=3e-6f*std::max(1.0f,std::abs(expected[i])));
                }
            }
#endif
        }
    }
}
void test_errors_and_ieee_results() {
    auto m=model({value(0,{1}),value(1,{1})},1,{op(0,OperatorKind::Scale,{0},{1},{},ScalePayload{ScaleSource::LiteralF32,bits(2)})});
    float maximum=std::numeric_limits<float>::max();
    equal(run(m,{{0,data({1},{maximum})}},{}),{std::numeric_limits<float>::infinity()});
    CHECK(std::holds_alternative<CompatibilityReport>(run(m,{{0,data({1},{std::numeric_limits<float>::infinity()})}},{})));
    m.operators={op(0,OperatorKind::Linear,{0},{1},{8},LinearPayload{})};m.tensors={tensor(8,{1,1})};
    equal(run(m,{{0,data({1},{maximum})}},{{8,data({1,1},{2})}}),{std::numeric_limits<float>::infinity()});
    m.operators[0].payload=AddPayload{};
    CHECK(std::holds_alternative<CompatibilityReport>(compile_semantic_program(m)));
    m.operators={op(0,OperatorKind::L2Normalize,{0},{1},{},L2NormalizePayload{0})};
    CHECK(std::holds_alternative<CompatibilityReport>(run(m,{{0,data({1},{0})}},{})));
    m.states.push_back({});CHECK(std::holds_alternative<CompatibilityReport>(compile_semantic_program(m)));
    m.states.clear();m.values[1].dimensions=shape({2});
    CHECK(std::holds_alternative<CompatibilityReport>(compile_semantic_program(m)));
}
}
int main() {
    test_linear_association();test_norms();test_activation_and_composition();
    test_embedding_and_binding();test_grouped_norm_and_shape_rejection();test_errors_and_ieee_results();test_stateful_rope_attention();
    return test_summary("test_semantic_program_compiler");
}
