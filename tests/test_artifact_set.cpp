#include <array>
#include <cerrno>
#include <cstring>
#include <dirent.h>
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
timespec g_original_times[2]{};

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

void replace_snapshot_payload(int directory_fd) {
    const int replacement_fd = openat(directory_fd, "replacement", O_WRONLY | O_CREAT | O_EXCL, 0600);
    CHECK(replacement_fd >= 0);
    if (replacement_fd >= 0) {
        CHECK(write(replacement_fd, "new", 3) == 3);
        close(replacement_fd);
        CHECK(renameat(directory_fd, "replacement", directory_fd, "payload") == 0);
    }
}

void mutate_restore_mtime_after_digest(int) {
    int writer = open(g_source_path.c_str(), O_WRONLY);
    CHECK(writer >= 0);
    if (writer >= 0) {
        CHECK(pwrite(writer, "new", 3, 0) == 3);
        CHECK(futimens(writer, g_original_times) == 0);
        close(writer);
    }
}

size_t snapshot_directory_count() {
    DIR* directory = opendir("/private/tmp");
    CHECK(directory != nullptr);
    if (!directory) return 0;
    size_t count = 0;
    constexpr char prefix[] = "laplace-artifact-snapshot-dir-";
    while (dirent* entry = readdir(directory)) {
        if (std::strncmp(entry->d_name, prefix, sizeof(prefix) - 1) == 0) ++count;
    }
    closedir(directory);
    return count;
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

    int mismatched_fd = open(path.c_str(), O_RDONLY);
    CHECK(mismatched_fd >= 0);
    errno = E2BIG;
    CHECK(!mapped.map_owned_fd(mismatched_fd, 4));
    CHECK(errno == EINVAL);
    errno = 0;
    CHECK(fcntl(mismatched_fd, F_GETFD) == -1 && errno == EBADF);
    unlink(path.c_str());
}

void test_fd_map_rejects_self_transfer_without_descriptor_reuse() {
    const std::string path = make_path(".self-transfer");
    write_bytes(path, "abc", 3);
    const int source_fd = open(path.c_str(), O_RDONLY);
    CHECK(source_fd >= 0);
    MappedFile mapped;
    CHECK(mapped.map_owned_fd(source_fd, 3));
    const int owned_fd = mapped.fd();
    errno = 0;
    CHECK(!mapped.map_owned_fd(owned_fd, 3));
    CHECK(errno == EINVAL);
    errno = 0;
    CHECK(fcntl(owned_fd, F_GETFD) >= 0);
    mapped.close();

    const int victim_fd = open(path.c_str(), O_RDONLY);
    CHECK(victim_fd >= 0);
    CHECK(victim_fd == owned_fd);
    errno = 0;
    CHECK(fcntl(victim_fd, F_GETFD) >= 0);
    close(victim_fd);
    unlink(path.c_str());
}

void test_bounded_read_advice_validates_before_os_calls() {
    const std::string path = make_path(".advice");
    write_bytes(path, "abcdef", 6);
    const int source_fd = open(path.c_str(), O_RDONLY);
    CHECK(source_fd >= 0);
    MappedFile mapped;
    CHECK(mapped.map_owned_fd(source_fd, 6));

    const std::array<MappedReadRange, 2> valid = {{{0, 2}, {4, 2}}};
    const MappedReadAdviceResult applied = mapped.advise_read_ranges(valid);
    CHECK(applied.requested_ranges == 2);
    CHECK(applied.requested_bytes == 4);
    CHECK(applied.memory_hint_applied);
#ifdef __APPLE__
    CHECK(applied.file_hint_applied);
#else
    CHECK(!applied.file_hint_applied);
#endif

    const std::array<MappedReadRange, 2> invalid = {{{0, 2}, {5, 2}}};
    const MappedReadAdviceResult rejected = mapped.advise_read_ranges(invalid);
    CHECK(!rejected.memory_hint_applied);
    CHECK(!rejected.file_hint_applied);
    CHECK(rejected.error == EINVAL);
    CHECK(rejected.requested_ranges == 0);
    CHECK(rejected.requested_bytes == 0);

    mapped.close();
    unlink(path.c_str());
}

void test_package_view_advice_uses_retained_owner() {
    const std::string path = make_path(".view-advice");
    write_bytes(path, "abcdef", 6);
    auto loaded = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    if (!std::holds_alternative<ArtifactSet>(loaded)) {
        unlink(path.c_str());
        return;
    }
    auto view_result = std::get<ArtifactSet>(std::move(loaded)).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(view_result));
    if (!std::holds_alternative<PackageView>(view_result)) {
        unlink(path.c_str());
        return;
    }
    const PackageView view = std::get<PackageView>(std::move(view_result));
    const std::array<MappedReadRange, 1> ranges = {{{1, 3}}};
    const MappedReadAdviceResult first = view.advise_read_ranges(ranges);
    const PackageView copy = view;
    const MappedReadAdviceResult second = copy.advise_read_ranges(ranges);
    CHECK(first.requested_ranges == 1);
    CHECK(first.requested_bytes == 3);
    CHECK(second.requested_ranges == 1);
    CHECK(second.requested_bytes == 3);
    CHECK(view.bytes().size() == 6);
    CHECK(copy.digest() == view.digest());
    unlink(path.c_str());
}

void test_snapshot_identity_mismatch_preserves_eio() {
    const std::string path = make_path(".identity-mismatch");
    write_bytes(path, "old", 3);
    const int source_fd = open(path.c_str(), O_RDONLY);
    CHECK(source_fd >= 0);
    const size_t before = snapshot_directory_count();
    MappedFile::set_test_force_snapshot_fallback(true);
    MappedFile::set_test_snapshot_hook(replace_snapshot_payload);
    errno = E2BIG;
    MappedFile mapped;
    CHECK(!mapped.map_snapshot_fd(source_fd, 3));
    CHECK(errno == EIO);
    CHECK(snapshot_directory_count() == before);
    MappedFile::set_test_snapshot_hook(nullptr);
    MappedFile::set_test_force_snapshot_fallback(false);
    errno = 0;
    CHECK(fcntl(source_fd, F_GETFD) == -1 && errno == EBADF);
    unlink(path.c_str());
}

#ifdef __APPLE__
void test_flagged_source_rejects_without_snapshot_leak() {
    const std::string path = make_path(".flagged");
    write_bytes(path, "abc", 3);
    for (unsigned long flag : {static_cast<unsigned long>(UF_IMMUTABLE),
                               static_cast<unsigned long>(UF_APPEND)}) {
        CHECK(chflags(path.c_str(), flag) == 0);
        const size_t before = snapshot_directory_count();
        const int source_fd = open(path.c_str(), O_RDONLY);
        CHECK(source_fd >= 0);
        MappedFile mapped;
        errno = 0;
        CHECK(!mapped.map_snapshot_fd(source_fd, 3));
        CHECK(errno == EPERM || errno == EACCES);
        CHECK(snapshot_directory_count() == before);
        auto artifact = ArtifactSet::load_single_file(path);
        CHECK(std::holds_alternative<CompatibilityReport>(artifact));
        if (const auto* report = std::get_if<CompatibilityReport>(&artifact)) {
            CHECK(report->code == CompatibilityError::PACKAGE_SNAPSHOT_UNAVAILABLE);
        }
        CHECK(snapshot_directory_count() == before);
        CHECK(chflags(path.c_str(), 0) == 0);
    }
    unlink(path.c_str());
}
#endif

void test_forced_disk_snapshot_is_read_only_and_bounded() {
    const std::string path = make_path(".fallback");
    write_bytes(path, "old", 3);

    int source_fd = open(path.c_str(), O_RDONLY);
    CHECK(source_fd >= 0);
    MappedFile::set_test_force_snapshot_fallback(true);
    MappedFile mapped;
    CHECK(mapped.map_snapshot_fd(source_fd, 3));
    MappedFile::set_test_force_snapshot_fallback(false);
    CHECK(mapped.snapshot_kind() == SnapshotKind::DiskCopy);
    CHECK(mapped.valid());
    const int retained_fd = mapped.fd();
    CHECK((fcntl(retained_fd, F_GETFD) & FD_CLOEXEC) != 0);
    errno = 0;
    CHECK(pwrite(retained_fd, "new", 3, 0) == -1);
    CHECK(errno == EBADF || errno == EINVAL);
    CHECK(std::memcmp(mapped.data(), "old", 3) == 0);
    mapped.close();
    errno = 0;
    CHECK(fcntl(retained_fd, F_GETFD) == -1 && errno == EBADF);
    unlink(path.c_str());

    const std::string large_path = make_path(".fallback-limit");
    const size_t too_large = MappedFile::snapshot_fallback_max_bytes_for_testing() + 1;
    int large_writer = open(large_path.c_str(), O_WRONLY | O_TRUNC);
    CHECK(large_writer >= 0);
    if (large_writer >= 0) {
        CHECK(ftruncate(large_writer, static_cast<off_t>(too_large)) == 0);
        close(large_writer);
    }
    int large_source_fd = open(large_path.c_str(), O_RDONLY);
    CHECK(large_source_fd >= 0);
    MappedFile::set_test_force_snapshot_fallback(true);
    errno = 0;
    CHECK(!mapped.map_snapshot_fd(large_source_fd, too_large));
    CHECK(errno == EFBIG);
    MappedFile::set_test_force_snapshot_fallback(false);
    errno = 0;
    CHECK(fcntl(large_source_fd, F_GETFD) == -1 && errno == EBADF);

    MappedFile::set_test_force_snapshot_fallback(true);
    auto large_artifact = ArtifactSet::load_single_file(large_path);
    MappedFile::set_test_force_snapshot_fallback(false);
    CHECK(std::holds_alternative<CompatibilityReport>(large_artifact));
    if (const auto* report = std::get_if<CompatibilityReport>(&large_artifact)) {
        CHECK(report->code == CompatibilityError::PACKAGE_SNAPSHOT_UNAVAILABLE);
        CHECK(report->detail.find("1 GiB") != std::string::npos);
    }
    unlink(large_path.c_str());

    const std::string artifact_path = make_path(".fallback-artifact");
    write_bytes(artifact_path, "old", 3);
    MappedFile::set_test_force_snapshot_fallback(true);
    auto loaded = ArtifactSet::load_single_file(artifact_path);
    MappedFile::set_test_force_snapshot_fallback(false);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    if (auto* artifacts = std::get_if<ArtifactSet>(&loaded)) {
        auto view = artifacts->view(ArtifactId{0});
        CHECK(std::holds_alternative<PackageView>(view));
        if (auto* package = std::get_if<PackageView>(&view)) {
            CHECK(std::memcmp(package->bytes().data(), "old", 3) == 0);
            CHECK(package->digest().hex() == "cba06b5736faf67e54b07b561eae94395e774c517a7d910a54369e1263ccfbd4");
        }
    }
    unlink(artifact_path.c_str());
}

void test_snapshot_descriptor_contract() {
    const std::string path = make_path(".descriptor");
    write_bytes(path, "old", 3);
    const int source_fd = open(path.c_str(), O_RDONLY);
    CHECK(source_fd >= 0);
    MappedFile mapped;
    MappedFile::set_test_force_snapshot_fallback(false);
    CHECK(mapped.map_snapshot_fd(source_fd, 3));
    CHECK(mapped.snapshot_kind() == SnapshotKind::Clone || mapped.snapshot_kind() == SnapshotKind::DiskCopy);
    const int retained_fd = mapped.fd();
    CHECK((fcntl(retained_fd, F_GETFD) & FD_CLOEXEC) != 0);
    errno = 0;
    CHECK(pwrite(retained_fd, "new", 3, 0) == -1);
    CHECK(errno == EBADF || errno == EINVAL);
    CHECK(std::memcmp(mapped.data(), "old", 3) == 0);
    mapped.close();
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

void test_post_load_equal_size_overwrite_does_not_mutate_snapshot() {
    const std::string path = make_path(".immutable");
    write_bytes(path, "old", 3);

    std::optional<PackageView> retained;
    {
        auto loaded = ArtifactSet::load_single_file(path);
        CHECK(std::holds_alternative<ArtifactSet>(loaded));
        if (auto* artifacts = std::get_if<ArtifactSet>(&loaded)) {
            auto view = artifacts->view(ArtifactId{0});
            CHECK(std::holds_alternative<PackageView>(view));
            if (auto* package = std::get_if<PackageView>(&view)) retained.emplace(*package);
        }
    }
    CHECK(retained.has_value());
    write_bytes(path, "new", 3);
    if (retained) {
        CHECK(std::memcmp(retained->bytes().data(), "old", 3) == 0);
        CHECK(retained->digest().hex() == "cba06b5736faf67e54b07b561eae94395e774c517a7d910a54369e1263ccfbd4");
    }
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

    ArtifactIdentityForTesting identity{1, 2, 3, 4, 5, 6, 7, 8};
    CHECK(artifact_identity_equal_for_testing(identity, identity));
    CHECK(!artifact_identity_equal_for_testing(identity, ArtifactIdentityForTesting{2, 2, 3, 4, 5, 6, 7, 8}));
    CHECK(!artifact_identity_equal_for_testing(identity, ArtifactIdentityForTesting{1, 3, 3, 4, 5, 6, 7, 8}));
    CHECK(!artifact_identity_equal_for_testing(identity, ArtifactIdentityForTesting{1, 2, 3, 4, 5, 9, 7, 8}));
    CHECK(!artifact_identity_equal_for_testing(identity, ArtifactIdentityForTesting{1, 2, 3, 4, 5, 6, 7, 9}));

    g_source_path = path;
    ArtifactSet::set_test_hook(mutate_source_after_digest);
    auto changed = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<CompatibilityReport>(changed));
    if (auto* report = std::get_if<CompatibilityReport>(&changed)) {
        CHECK(report->code == CompatibilityError::PACKAGE_SOURCE_CHANGED);
    }
    ArtifactSet::set_test_hook(nullptr);

    write_bytes(path, "old", 3);
    struct stat original_stat {};
    CHECK(stat(path.c_str(), &original_stat) == 0);
    g_original_times[0] = original_stat.st_atimespec;
    g_original_times[1] = original_stat.st_mtimespec;
    g_source_path = path;
    ArtifactSet::set_test_hook(mutate_restore_mtime_after_digest);
    auto restored_mtime_changed = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<CompatibilityReport>(restored_mtime_changed));
    if (auto* report = std::get_if<CompatibilityReport>(&restored_mtime_changed)) {
        CHECK(report->code == CompatibilityError::PACKAGE_SOURCE_CHANGED);
    }
    ArtifactSet::set_test_hook(nullptr);

    write_bytes(path, "old", 3);
    g_source_path = path;
    ArtifactSet::set_test_hook(replace_directory_entry_after_digest);
    auto retained_descriptor = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<CompatibilityReport>(retained_descriptor));
    if (const auto* report = std::get_if<CompatibilityReport>(&retained_descriptor)) {
        CHECK(report->code == CompatibilityError::PACKAGE_SOURCE_CHANGED);
    }
    ArtifactSet::set_test_hook(nullptr);
    unlink(link.c_str());
    unlink(path.c_str());
    rmdir(directory.c_str());
}

void test_owned_blob_is_immutable_and_self_contained() {
    std::array<uint8_t, 4> source = {0x10, 0x20, 0x30, 0x40};
    auto result = ArtifactSet::make_owned_blob(ArtifactId{37}, ArtifactRole::Shard,
                                               std::span<const uint8_t>(source));
    CHECK(std::holds_alternative<PackageView>(result));
    if (!std::holds_alternative<PackageView>(result)) return;
    PackageView retained = std::get<PackageView>(result);
    source[0] = 0xff;
    result = CompatibilityReport{};
    CHECK(retained.artifact_id() == ArtifactId{37});
    CHECK(retained.role() == ArtifactRole::Shard);
    CHECK(retained.bytes().size() == source.size());
    CHECK(retained.bytes()[0] == 0x10);
    CHECK(retained.digest().hex() ==
          "f4e3f0b04771c047e227c9ecaba65d3fe2fd0e1eee0a7552b956d1a7c535a7cf");

    auto empty = ArtifactSet::make_owned_blob(ArtifactId{37}, ArtifactRole::Shard, {});
    CHECK(std::holds_alternative<CompatibilityReport>(empty));
    if (const auto* report = std::get_if<CompatibilityReport>(&empty)) {
        CHECK(report->code == CompatibilityError::PACKAGE_BOUNDS_INVALID);
    }
    auto bad_id = ArtifactSet::make_owned_blob(ArtifactId{UINT32_MAX}, ArtifactRole::Shard,
                                                std::span<const uint8_t>(source));
    CHECK(std::holds_alternative<CompatibilityReport>(bad_id));
    if (const auto* report = std::get_if<CompatibilityReport>(&bad_id)) {
        CHECK(report->code == CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED);
    }
    auto bad_role = ArtifactSet::make_owned_blob(
        ArtifactId{37}, static_cast<ArtifactRole>(99),
        std::span<const uint8_t>(source));
    CHECK(std::holds_alternative<CompatibilityReport>(bad_role));
    if (const auto* report = std::get_if<CompatibilityReport>(&bad_role)) {
        CHECK(report->code == CompatibilityError::PACKAGE_GRAPH_UNSUPPORTED);
    }
}

void test_package_subview_retains_owner_and_bounds() {
    const std::string path = make_path(".subview");
    write_bytes(path, "0123456789", 10);
    auto loaded = ArtifactSet::load_single_file(path);
    CHECK(std::holds_alternative<ArtifactSet>(loaded));
    if (!std::holds_alternative<ArtifactSet>(loaded)) {
        unlink(path.c_str());
        return;
    }
    auto source = std::get<ArtifactSet>(std::move(loaded)).view(ArtifactId{0});
    CHECK(std::holds_alternative<PackageView>(source));
    if (!std::holds_alternative<PackageView>(source)) {
        unlink(path.c_str());
        return;
    }
    auto sliced = ArtifactSet::make_subview(
        std::get<PackageView>(source), ArtifactId{9}, ArtifactRole::Shard, 3, 4);
    CHECK(std::holds_alternative<PackageView>(sliced));
    source = CompatibilityReport{};
    loaded = CompatibilityReport{};
    if (const auto* view = std::get_if<PackageView>(&sliced)) {
        CHECK(view->artifact_id() == ArtifactId{9});
        CHECK(view->role() == ArtifactRole::Shard);
        CHECK(std::string_view(reinterpret_cast<const char*>(view->bytes().data()),
                               view->bytes().size()) == "3456");
        const std::array<MappedReadRange, 1> valid = {{{1, 2}}};
        const auto advised = view->advise_read_ranges(valid);
        CHECK(advised.requested_ranges == 1);
        CHECK(advised.requested_bytes == 2);
        const std::array<MappedReadRange, 1> invalid = {{{3, 2}}};
        const auto rejected = view->advise_read_ranges(invalid);
        CHECK(rejected.error == EINVAL);
        CHECK(rejected.requested_ranges == 0);
    }
    auto empty = ArtifactSet::make_subview(
        std::get<PackageView>(sliced), ArtifactId{10}, ArtifactRole::Shard, 0, 0);
    CHECK(std::holds_alternative<CompatibilityReport>(empty));
    auto outside = ArtifactSet::make_subview(
        std::get<PackageView>(sliced), ArtifactId{10}, ArtifactRole::Shard, 3, 2);
    CHECK(std::holds_alternative<CompatibilityReport>(outside));
    unlink(path.c_str());
}

} // namespace

int main() {
    test_fd_map_consumes_fd();
    test_fd_map_rejects_self_transfer_without_descriptor_reuse();
    test_bounded_read_advice_validates_before_os_calls();
    test_package_view_advice_uses_retained_owner();
    test_snapshot_identity_mismatch_preserves_eio();
#ifdef __APPLE__
    test_flagged_source_rejects_without_snapshot_leak();
#endif
    test_forced_disk_snapshot_is_read_only_and_bounded();
    test_snapshot_descriptor_contract();
    test_checked_owner_and_graph_rejection();
    test_post_load_equal_size_overwrite_does_not_mutate_snapshot();
    test_multifile_graph_has_stable_ids_and_digests();
    test_rejected_graph_releases_candidate_mapping();
    test_source_rejections_and_identity();
    test_owned_blob_is_immutable_and_self_contained();
    test_package_subview_retains_owner_and_bounds();
    return test_summary("test_artifact_set");
}
