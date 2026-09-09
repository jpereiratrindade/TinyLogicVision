#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace tinyvision {

enum class NodeType {
    SOURCE,
    ROI,
    PATCH,
    DATASET,
    TRAINING_RUN,
    MODEL,
    CLASSIFICATION_RUN,
    ARTIFACT,
    EXPERIMENT
};

inline std::string node_type_to_string(NodeType t) {
    switch (t) {
        case NodeType::SOURCE: return "Source";
        case NodeType::ROI: return "ROI";
        case NodeType::PATCH: return "Patch";
        case NodeType::DATASET: return "Dataset";
        case NodeType::TRAINING_RUN: return "TrainingRun";
        case NodeType::MODEL: return "Model";
        case NodeType::CLASSIFICATION_RUN: return "ClassificationRun";
        case NodeType::ARTIFACT: return "Artifact";
        case NodeType::EXPERIMENT: return "Experiment";
    }
    return "Unknown";
}

inline NodeType string_to_node_type(const std::string& s) {
    if (s == "Source") return NodeType::SOURCE;
    if (s == "ROI") return NodeType::ROI;
    if (s == "Patch") return NodeType::PATCH;
    if (s == "Dataset") return NodeType::DATASET;
    if (s == "TrainingRun") return NodeType::TRAINING_RUN;
    if (s == "Model") return NodeType::MODEL;
    if (s == "ClassificationRun") return NodeType::CLASSIFICATION_RUN;
    if (s == "Artifact") return NodeType::ARTIFACT;
    return NodeType::EXPERIMENT;
}

enum class EdgeType {
    DERIVED_FROM,
    SAMPLED_FROM,
    MEMBER_OF,
    TRAINED_ON,
    PRODUCED_BY,
    EVALUATED_ON,
    SUCCESSOR_EXPERIMENT_OF
};

inline std::string edge_type_to_string(EdgeType t) {
    switch (t) {
        case EdgeType::DERIVED_FROM: return "derived_from";
        case EdgeType::SAMPLED_FROM: return "sampled_from";
        case EdgeType::MEMBER_OF: return "member_of";
        case EdgeType::TRAINED_ON: return "trained_on";
        case EdgeType::PRODUCED_BY: return "produced_by";
        case EdgeType::EVALUATED_ON: return "evaluated_on";
        case EdgeType::SUCCESSOR_EXPERIMENT_OF: return "successor_experiment_of";
    }
    return "related_to";
}

inline EdgeType string_to_edge_type(const std::string& s) {
    if (s == "derived_from") return EdgeType::DERIVED_FROM;
    if (s == "sampled_from") return EdgeType::SAMPLED_FROM;
    if (s == "member_of") return EdgeType::MEMBER_OF;
    if (s == "trained_on") return EdgeType::TRAINED_ON;
    if (s == "produced_by") return EdgeType::PRODUCED_BY;
    if (s == "evaluated_on") return EdgeType::EVALUATED_ON;
    return EdgeType::SUCCESSOR_EXPERIMENT_OF;
}

struct ProvenanceNode {
    std::string id;
    NodeType type{NodeType::EXPERIMENT};
    std::string name;
    std::string hash_sha256;
    std::map<std::string, std::string> metadata;
};

struct ProvenanceEdge {
    std::string source_id;
    std::string target_id;
    EdgeType type{EdgeType::DERIVED_FROM};
    std::string timestamp;
    std::map<std::string, std::string> metadata;
};

class ProvenanceGraph {
public:
    void add_node(ProvenanceNode node);
    void add_edge(ProvenanceEdge edge);

    const ProvenanceNode* find_node(const std::string& id) const;
    std::vector<ProvenanceEdge> get_incoming_edges(const std::string& target_id) const;
    std::vector<ProvenanceEdge> get_outgoing_edges(const std::string& source_id) const;

    void export_json(const std::filesystem::path& path) const;
    void export_jsonl(const std::filesystem::path& path) const;
    static ProvenanceGraph load_json(const std::filesystem::path& path);

    const std::vector<ProvenanceNode>& nodes() const noexcept { return nodes_; }
    const std::vector<ProvenanceEdge>& edges() const noexcept { return edges_; }

private:
    std::vector<ProvenanceNode> nodes_;
    std::vector<ProvenanceEdge> edges_;
};

} // namespace tinyvision
