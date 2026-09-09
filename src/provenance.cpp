#include "tinyvision/provenance.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace tinyvision {
namespace {

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': escaped << "\\\""; break;
        case '\\': escaped << "\\\\"; break;
        case '\b': escaped << "\\b"; break;
        case '\f': escaped << "\\f"; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default:
            if (ch < 0x20) {
                escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(ch) << std::dec;
            } else {
                escaped << static_cast<char>(ch);
            }
        }
    }
    return escaped.str();
}

} // namespace

void ProvenanceGraph::add_node(ProvenanceNode node) {
    if (find_node(node.id) != nullptr) {
        return; // Idempotent node addition
    }
    nodes_.push_back(std::move(node));
}

void ProvenanceGraph::add_edge(ProvenanceEdge edge) {
    edges_.push_back(std::move(edge));
}

const ProvenanceNode* ProvenanceGraph::find_node(const std::string& id) const {
    for (const auto& node : nodes_) {
        if (node.id == id) return &node;
    }
    return nullptr;
}

std::vector<ProvenanceEdge> ProvenanceGraph::get_incoming_edges(const std::string& target_id) const {
    std::vector<ProvenanceEdge> res;
    for (const auto& e : edges_) {
        if (e.target_id == target_id) res.push_back(e);
    }
    return res;
}

std::vector<ProvenanceEdge> ProvenanceGraph::get_outgoing_edges(const std::string& source_id) const {
    std::vector<ProvenanceEdge> res;
    for (const auto& e : edges_) {
        if (e.source_id == source_id) res.push_back(e);
    }
    return res;
}

void ProvenanceGraph::export_json(const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open file for writing: " + path.string());

    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"nodes\": [\n";

    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        const auto& n = nodes_[i];
        out << "    {\n"
            << "      \"id\": \"" << json_escape(n.id) << "\",\n"
            << "      \"type\": \"" << node_type_to_string(n.type) << "\",\n"
            << "      \"name\": \"" << json_escape(n.name) << "\",\n"
            << "      \"hash_sha256\": \"" << json_escape(n.hash_sha256) << "\",\n"
            << "      \"metadata\": {\n";
        std::size_t m_idx = 0;
        for (const auto& [k, v] : n.metadata) {
            out << "        \"" << json_escape(k) << "\": \"" << json_escape(v) << "\"" << (++m_idx < n.metadata.size() ? ",\n" : "\n");
        }
        out << "      }\n"
            << "    }" << (i + 1 < nodes_.size() ? ",\n" : "\n");
    }

    out << "  ],\n"
        << "  \"edges\": [\n";

    for (std::size_t i = 0; i < edges_.size(); ++i) {
        const auto& e = edges_[i];
        out << "    {\n"
            << "      \"source\": \"" << json_escape(e.source_id) << "\",\n"
            << "      \"target\": \"" << json_escape(e.target_id) << "\",\n"
            << "      \"type\": \"" << edge_type_to_string(e.type) << "\",\n"
            << "      \"timestamp\": \"" << json_escape(e.timestamp) << "\",\n"
            << "      \"metadata\": {\n";
        std::size_t m_idx = 0;
        for (const auto& [k, v] : e.metadata) {
            out << "        \"" << json_escape(k) << "\": \"" << json_escape(v) << "\"" << (++m_idx < e.metadata.size() ? ",\n" : "\n");
        }
        out << "      }\n"
            << "    }" << (i + 1 < edges_.size() ? ",\n" : "\n");
    }

    out << "  ]\n"
        << "}\n";
}

void ProvenanceGraph::export_jsonl(const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open file for writing: " + path.string());

    for (const auto& n : nodes_) {
        out << "{\"kind\":\"node\",\"id\":\"" << json_escape(n.id) << "\",\"type\":\"" << node_type_to_string(n.type)
            << "\",\"name\":\"" << json_escape(n.name) << "\",\"hash_sha256\":\"" << json_escape(n.hash_sha256) << "\"}\n";
    }
    for (const auto& e : edges_) {
        out << "{\"kind\":\"edge\",\"source\":\"" << json_escape(e.source_id) << "\",\"target\":\"" << json_escape(e.target_id)
            << "\",\"type\":\"" << edge_type_to_string(e.type) << "\",\"timestamp\":\"" << json_escape(e.timestamp) << "\"}\n";
    }
}

} // namespace tinyvision
