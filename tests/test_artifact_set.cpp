#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "artifact_set.h"
#include "mmap.h"
#include "test_util.h"

using namespace Laplace;

namespace {

std::string g_source_path;

std::string make_path(const char* suffix) {
    char path[] = "/private/tmp/laplace-artifact-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd >= 0) close(fd);
    std::string result = std::string(path) + suffix;
    rename(path, result.c_str());
    return result;
}

void write_bytes(const std::string& path, const char* bytes, size_t count) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0);
    if (fd < 0) return;
    CHECK(write(fd, bytes, count) == static_cast<ssize_t>(count));
    close(fd);
}

void mutate_source_after_digest(int) {
    int writer = open(g_source_path.c_str(), O_WRONLY);
    CHECK(writer >= 0);
    if (writer >= 0) {
        CHECK(ftruncate(writer, 4) == 0);
        close(writer);
    }
}

void replace_directory_entry_after_digest(int) {
    std::string replacement = g_source_path + ".replacement";
    write_bytes(replacement, "new", 3);
    CHECK(rename(replacement.c_str(), g_source_path.c_str()) == 0);
}

void test_fd_map_consumes_fd() {
    std::string path = make_path(".map");
    write_bytes(path, "abc", 3);

    int success_fd = open(path.c_str(), O_RDONLY);
    CHECK(success_fd >= 0);
    MappedFile mapped;
    CHECK(mapped.map_owned_fd(success_fd, 3));
    CHECK(mapped.valid());
    CHECK(std::memcmp(mapped.data(), "abc", 3) == 0);
    mapped.close();
    errno = 0;
    CHECK(fcntl(success_fd, F_GETFD) == -1 && errno == EBADF);

    int failure_fd = open(path.c_str(), O_RDONLY);
    CHECK(failure_fd >= 0);
    CHECK(!mapped.map_owned_fd(failure_fd, 0));
    errno = 0;
    CHECK(fcntl(failure_fd, F_GETFD) == -1 && errno == EBADF);
    unlink(path.c_str());
}

void test_checked_owner_and_graph_rejection() {
    std::string path = make_path(".gguf");
    write_bytes(path, "abc", 3);
    std::optional<PackageView> retained_view;
    {
        auto loaded = ArtifactSet::load_single_file(path);
        CHECK(std::holds_alternative<ArtifactSet>(loaded));
        if (auto* artifacts = std::get_if<ArtifactSet>(&loaded)) {
            auto view = artifacts->view(ArtifactId{0});
            CHECK(std::holds_alternative<PackageView>(view));
            if (auto* package = std::get_if<PackageView>(&view)) retained_view.emplace(*package);
        }
    }
    CHECK(retained_view.has_value());
    if (!retained_view) {
        unlink(path.c_str());
        return;
    }
    CHECK(retained_view->bytes().size() == 3);
    CHECK(std::memcmp(retained_view->bytes().data(), "abc", 3) == 0);
    CHECK(retained_view->digest().hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    unlink(path.c_str());
}

void test_multifile_graph_has_stable_ids_and_digests() {
    const std::string root = make_path(".root");
    const std::string shard = make_path(".shard");
    const std::string sidecar = make_path(".sidecar");
    write_bytes(root, "root", 4);
    write_bytes(shard, "shard", 5);
    write_bytes(sidecar, "sidecar", 7);

    const std::array<ArtifactSource, 3> sources = {{
        {root, ArtifactRole::Primary, ArtifactId{4}},
        {shard, ArtifactRole::Shard, ArtifactId{9}},
        {sidecar, ArtifactRole::Sidecar, ArtifactId{12}},
    }};
    auto loaded = ArtifactSet::load_graph(sources);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    if (auto* artifacts = std::get_if<ArtifactSet>(&loaded)) {
        auto root_view = artifacts->view(ArtifactId{4});
        auto shard_view = artifacts->view(ArtifactId{9});
        auto sidecar_view = artifacts->view(ArtifactId{12});
        CHECK(std::holds_alternative<PackageView>(root_view));
        CHECK(std::holds_alternative<PackageView>(shard_view));
        CHECK(std::holds_alternative<PackageView>(sidecar_view));
        if (auto* package = std::get_if<PackageView>(&root_view)) {
            CHECK(package->artifact_id() == ArtifactId{4});
            CHECK(std::memcmp(package->bytes().data(), "root", 4) == 0);
            CHECK(package->digest().hex() == "4813494d137e1631bba301d5acab6e7bb7aa74ce1185d456565ef51d737677b2");
        }
        if (auto* package = std::get_if<PackageView>(&shard_view)) {
            CHECK(package->artifact_id() == ArtifactId{9});
            CHECK(std::memcmp(package->bytes().data(), "shard", 5) == 0);
        }
        if (auto* package = std::get_if<PackageView>(&sidecar_view)) {
            CHECK(package->artifact_id() == ArtifactId{12});
            CHECK(std::memcmp(package->bytes().data(), "sidecar", 7) == 0);
        }
        auto missing = artifacts->view(ArtifactId{1});
        CHECK(std::holds_alternative<CompatibilityReport>(missing));
        if (const auto* report = std::get_if<CompatibilityReport>(&missing)) {
            CHECK(report->code == CompatibilityError::IR_REFERENCE_INVALID);
        }
    }

    const std::array<ArtifactSource, 2> duplicate_member = {{
        {root, ArtifactRole::Primary, ArtifactId{4}},
        {shard, ArtifactRole::Shard, ArtifactId{4}},
    }};
    auto duplicate_member_result = ArtifactSet::load_graph(duplicate_member);
    CHECK(std::holds_alternative<CompatibilityReport>(duplicate_member_result));
    if (const auto* report = std::get_if<CompatibilityReport>(&duplicate_member_result)) {
        CHECK(report->code == CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED);
    }

    const std::array<ArtifactSource, 2> duplicate_root = {{
        {root, ArtifactRole::Primary, ArtifactId{4}},
        {shard, ArtifactRole::Primary, ArtifactId{9}},
    }};
    auto duplicate_root_result = ArtifactSet::load_graph(duplicate_root);
    CHECK(std::holds_alternative<CompatibilityReport>(duplicate_root_result));
    if (const auto* report = std::get_if<CompatibilityReport>(&duplicate_root_result)) {
        CHECK(report->code == CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED);
    }

    const std::array<ArtifactSource, 2> duplicate_source = {{
        {root, ArtifactRole::Primary, ArtifactId{4}},
        {root, ArtifactRole::Shard, ArtifactId{9}},
    }};
    auto duplicate_source_result = ArtifactSet::load_graph(duplicate_source);
    CHECK(std::holds_alternative<CompatibilityReport>(duplicate_source_result));
    if (const auto* report = std::get_if<CompatibilityReport>(&duplicate_source_result)) {
        CHECK(report->code == CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED);
    }

    auto first = ArtifactSet::load_single_file(root);
    write_bytes(root, "next", 4);
    auto changed = ArtifactSet::load_single_file(root);
    CHECK(std::holds_alternative<ArtifactSet>(first));
    CHECK(std::holds_alternative<ArtifactSet>(changed));
    if (auto* before = std::get_if<ArtifactSet>(&first)) {
        auto before_view = before->view(ArtifactId{0});
        if (auto* package = std::get_if<PackageView>(&before_view)) {
            if (auto* after = std::get_if<ArtifactSet>(&changed)) {
                auto after_view = after->view(ArtifactId{0});
                if (auto* changed_package = std::get_if<PackageView>(&after_view)) {
                    CHECK(package->digest() != changed_package->digest());
                }
            }
        }
    }

    const std::string copy = make_path(".copy");
    write_bytes(copy, "next", 4);
    auto copied = ArtifactSet::load_single_file(copy);
    CHECK(std::holds_alternative<ArtifactSet>(changed));
    CHECK(std::holds_alternative<ArtifactSet>(copied));
    if (auto* changed_set = std::get_if<ArtifactSet>(&changed)) {
        if (auto* copied_set = std::get_if<ArtifactSet>(&copied)) {
            auto changed_view = changed_set->view(ArtifactId{0});
            auto copied_view = copied_set->view(ArtifactId{0});
            if (auto* changed_package = std::get_if<PackageView>(&changed_view)) {
                if (auto* copied_package = std::get_if<PackageView>(&copied_view)) {
                    CHECK(changed_package->digest() == copied_package->digest());
                }
            }
        }
    }

    unlink(root.c_str());
    unlink(shard.c_str());
    unlink(sidecar.c_str());
    unlink(copy.c_str());
}

void test_rejected_graph_releases_candidate_mapping() {
    const std::string root = make_path(".candidate");
    write_bytes(root, "candidate", 9);
    CHECK(ArtifactSet::live_mapping_count_for_testing() == 0);
    const std::array<ArtifactSource, 2> duplicate_source = {{
        {root, ArtifactRole::Primary, ArtifactId{4}},
        {root, ArtifactRole::Shard, ArtifactId{9}},
    }};
    auto rejected = ArtifactSet::load_graph(duplicate_source);
    CHECK(std::holds_alternative<CompatibilityReport>(rejected));
    CHECK(ArtifactSet::live_mapping_count_for_testing() == 0);
    unlink(root.c_str());
}

void test_source_rejections_and_identity() {
    std::string path = make_path(".source");
    write_bytes(path, "abc", 3);
    std::string link = path + ".link";
    CHECK(symlink(path.c_str(), link.c_str()) == 0);
    auto symlink_result = ArtifactSet::load_single_file(link);
    CHECK(std::holds_alternative<CompatibilityReport>(symlink_result));
    if (auto* report = std::get_if<CompatibilityReport>(&symlink_result)) {
        CHECK(report->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);
    }

    std::string directory = make_path(".directory");
    unlink(directory.c_str());
    CHECK(mkdir(directory.c_str(), 0700) == 0);
    auto directory_result = ArtifactSet::load_single_file(directory);
    CHECK(std::holds_alternative<CompatibilityReport>(directory_result));
    if (auto* report = std::get_if<CompatibilityReport>(&directory_result)) {
        CHECK(report->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);
    }

    ArtifactIdentityForTesting identity{1, 2, 3, 4, 5};
    CHECK(artifact_identity_equal_for_testing(identity, identity));
    CHECK(!artifact_identity_equal_for_testing(identity, ArtifactIdentityForTesting{2, 2, 3, 4, 5}));
    CHECK(!artifact_identity_equal_for_testing(identity, ArtifactIdentityForTesting{1, 3, 3, 4, 5}));

    g_source_path = path;
    ArtifactSet::set_test_hook(mutate_source_after_digest);
    auto changed = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<CompatibilityReport>(changed));
    if (auto* report = std::get_if<CompatibilityReport>(&changed)) {
        CHECK(report->code == CompatibilityError::PACKAGE_SOURCE_CHANGED);
    }
    ArtifactSet::set_test_hook(nullptr);

    write_bytes(path, "old", 3);
    g_source_path = path;
    ArtifactSet::set_test_hook(replace_directory_entry_after_digest);
    auto retained_descriptor = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(retained_descriptor));
    if (auto* artifacts = std::get_if<ArtifactSet>(&retained_descriptor)) {
        auto view = artifacts->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            CHECK(std::memcmp(package->bytes().data(), "old", 3) == 0);
        }
    }
    ArtifactSet::set_test_hook(nullptr);
    unlink(link.c_str());
    unlink(path.c_str());
    rmdir(directory.c_str());
}

} // namespace

int main() {
    test_fd_map_consumes_fd();
    test_checked_owner_and_graph_rejection();
    test_multifile_graph_has_stable_ids_and_digests();
    test_rejected_graph_releases_candidate_mapping();
    test_source_rejections_and_identity();
    return test_summary("test_artifact_set");
}
