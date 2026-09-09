#include "tinyvision/provenance.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << '\n';
    std::exit(1);
}

} // namespace

int main() {
    std::cout << "Running provenance_test...\n";

    const auto tmp_dir = std::filesystem::temp_directory_path() / "tv_provenance_test";
    std::filesystem::create_directories(tmp_dir);

    tinyvision::ProvenanceGraph graph;

    // 1. Create Nodes
    graph.add_node({"exp-01", tinyvision::NodeType::EXPERIMENT, "TV-01A Generalization Observatory", "hash_exp1", {}});
    graph.add_node({"exp-02", tinyvision::NodeType::EXPERIMENT, "TV-01B Spatial Coverage", "hash_exp2", {}});
    graph.add_node({"src-image-01", tinyvision::NodeType::SOURCE, "sentinel_scene.png", "hash_src1", {{"crs", "EPSG:32722"}}});
    graph.add_node({"roi-floresta-01", tinyvision::NodeType::ROI, "ROI Floresta 1", "hash_roi1", {{"class", "floresta"}}});
    graph.add_node({"patch-001", tinyvision::NodeType::PATCH, "Patch #001", "hash_patch1", {{"origin_x", "10"}, {"origin_y", "20"}}});
    graph.add_node({"dataset-v1", tinyvision::NodeType::DATASET, "Dataset Sentinel RGB v1", "hash_ds1", {{"split_count", "3"}}});
    graph.add_node({"train-run-01", tinyvision::NodeType::TRAINING_RUN, "Run 2026-09-09 Training", "hash_tr1", {{"lr", "0.025"}}});
    graph.add_node({"model-01", tinyvision::NodeType::MODEL, "model_v2.tlv", "hash_model1", {{"modality", "RGB"}}});
    graph.add_node({"class-run-01", tinyvision::NodeType::CLASSIFICATION_RUN, "Dense Run #01", "hash_crun1", {{"stride", "1"}}});
    graph.add_node({"art-classmap", tinyvision::NodeType::ARTIFACT, "class_map.png", "hash_art1", {{"format", "png"}}});

    // 2. Create Edges
    graph.add_edge({"exp-02", "exp-01", tinyvision::EdgeType::SUCCESSOR_EXPERIMENT_OF, "2026-09-09T10:00:00Z", {}});
    graph.add_edge({"roi-floresta-01", "src-image-01", tinyvision::EdgeType::DERIVED_FROM, "2026-09-09T10:05:00Z", {}});
    graph.add_edge({"patch-001", "roi-floresta-01", tinyvision::EdgeType::SAMPLED_FROM, "2026-09-09T10:06:00Z", {}});
    graph.add_edge({"patch-001", "dataset-v1", tinyvision::EdgeType::MEMBER_OF, "2026-09-09T10:07:00Z", {{"split", "TRAIN"}}});
    graph.add_edge({"train-run-01", "dataset-v1", tinyvision::EdgeType::TRAINED_ON, "2026-09-09T10:10:00Z", {}});
    graph.add_edge({"model-01", "train-run-01", tinyvision::EdgeType::PRODUCED_BY, "2026-09-09T10:12:00Z", {}});
    graph.add_edge({"class-run-01", "model-01", tinyvision::EdgeType::EVALUATED_ON, "2026-09-09T10:15:00Z", {}});
    graph.add_edge({"art-classmap", "class-run-01", tinyvision::EdgeType::PRODUCED_BY, "2026-09-09T10:16:00Z", {}});

    // 3. Lineage Query: From which ROI and Source did patch-001 come?
    const auto patch_edges = graph.get_outgoing_edges("patch-001");
    std::string roi_id;
    for (const auto& e : patch_edges) {
        if (e.type == tinyvision::EdgeType::SAMPLED_FROM) {
            roi_id = e.target_id;
        }
    }
    if (roi_id != "roi-floresta-01") {
        fail("Lineage failed: patch-001 ROI origin mismatch");
    }

    const auto roi_edges = graph.get_outgoing_edges(roi_id);
    std::string src_id;
    for (const auto& e : roi_edges) {
        if (e.type == tinyvision::EdgeType::DERIVED_FROM) {
            src_id = e.target_id;
        }
    }
    if (src_id != "src-image-01") {
        fail("Lineage failed: roi-floresta-01 source image mismatch");
    }

    // 4. Lineage Query: Which experiment succeeded exp-01?
    const auto exp_edges = graph.get_incoming_edges("exp-01");
    if (exp_edges.empty() || exp_edges[0].source_id != "exp-02" ||
        exp_edges[0].type != tinyvision::EdgeType::SUCCESSOR_EXPERIMENT_OF) {
        fail("Lineage failed: successor experiment mismatch");
    }

    // 5. Test JSON and JSONL exports
    const auto json_path = tmp_dir / "provenance.json";
    const auto jsonl_path = tmp_dir / "evidence.jsonl";

    graph.export_json(json_path);
    graph.export_jsonl(jsonl_path);

    if (!std::filesystem::exists(json_path) || std::filesystem::file_size(json_path) == 0) {
        fail("provenance.json export failed");
    }
    if (!std::filesystem::exists(jsonl_path) || std::filesystem::file_size(jsonl_path) == 0) {
        fail("evidence.jsonl export failed");
    }

    // Cleanup
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);

    std::cout << "PASS: provenance_test (nodes, edges, RIT lineage queries, provenance.json/jsonl)\n";
    return 0;
}
