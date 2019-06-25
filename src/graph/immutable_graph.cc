/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/immutable_graph.cc
 * \brief DGL immutable graph index implementation
 */

#include <dgl/immutable_graph.h>
#include <string.h>
#include <bitset>
#include <numeric>
#include <tuple>

#include "../c_api_common.h"

namespace dgl {
namespace {
/*!
 * \brief A hashmap that maps each ids in the given array to new ids starting from zero.
 */
class IdHashMap {
 public:
  // Construct the hashmap using the given id arrays.
  // The id array could contain duplicates.
  explicit IdHashMap(IdArray ids): filter_(kFilterSize, false) {
    const dgl_id_t* ids_data = static_cast<dgl_id_t*>(ids->data);
    const int64_t len = ids->shape[0];
    dgl_id_t newid = 0;
    for (int64_t i = 0; i < len; ++i) {
      const dgl_id_t id = ids_data[i];
      if (!Contains(id)) {
        oldv2newv_[id] = newid++;
        filter_[id & kFilterMask] = true;
      }
    }
  }

  // Return true if the given id is contained in this hashmap.
  bool Contains(dgl_id_t id) const {
    return filter_[id & kFilterMask] && oldv2newv_.count(id);
  }

  // Return the new id of the given id. If the given id is not contained
  // in the hash map, returns the default_val instead.
  dgl_id_t Map(dgl_id_t id, dgl_id_t default_val) const {
    if (filter_[id & kFilterMask]) {
      auto it = oldv2newv_.find(id);
      return (it == oldv2newv_.end()) ? default_val : it->second;
    } else {
      return default_val;
    }
  }

 private:
  static constexpr int32_t kFilterMask = 0xFFFFFF;
  static constexpr int32_t kFilterSize = kFilterMask + 1;
  // This bitmap is used as a bloom filter to remove some lookups.
  // Hashtable is very slow. Using bloom filter can significantly speed up lookups.
  std::vector<bool> filter_;
  // The hashmap from old vid to new vid
  std::unordered_map<dgl_id_t, dgl_id_t> oldv2newv_;
};

struct PairHash {
  template <class T1, class T2>
  std::size_t operator() (const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

std::tuple<IdArray, IdArray, IdArray> MapFromSharedMemory(
  const std::string &shared_mem_name, int64_t num_verts, int64_t num_edges, bool is_create) {
#ifndef _WIN32
  const int64_t file_size = (num_verts + 1 + num_edges * 2) * sizeof(dgl_id_t);

  IdArray sm_array = IdArray::EmptyShared(
      shared_mem_name, {file_size}, DLDataType{kDLInt, 8, 1}, DLContext{kDLCPU, 0}, is_create);
  // Create views from the shared memory array. Note that we don't need to save
  //   the sm_array because the refcount is maintained by the view arrays.
  IdArray indptr = sm_array.CreateView({num_verts + 1}, DLDataType{kDLInt, 64, 1});
  IdArray indices = sm_array.CreateView({num_edges}, DLDataType{kDLInt, 64, 1},
      (num_verts + 1) * sizeof(dgl_id_t));
  IdArray edge_ids = sm_array.CreateView({num_edges}, DLDataType{kDLInt, 64, 1},
      (num_verts + 1 + num_edges) * sizeof(dgl_id_t));
  return std::make_tuple(indptr, indices, edge_ids);
#else
  LOG(FATAL) << "CSR graph doesn't support shared memory in Windows yet";
  return {};
#endif  // _WIN32
}
}  // namespace

//////////////////////////////////////////////////////////
//
// CSR graph implementation
//
//////////////////////////////////////////////////////////

CSR::CSR(int64_t num_vertices, int64_t num_edges, bool is_multigraph)
  : is_multigraph_(is_multigraph) {
  indptr_ = NewIdArray(num_vertices + 1);
  indices_ = NewIdArray(num_edges);
  edge_ids_ = NewIdArray(num_edges);
}

CSR::CSR(IdArray indptr, IdArray indices, IdArray edge_ids)
  : indptr_(indptr), indices_(indices), edge_ids_(edge_ids) {
  CHECK(IsValidIdArray(indptr));
  CHECK(IsValidIdArray(indices));
  CHECK(IsValidIdArray(edge_ids));
  CHECK_EQ(indices->shape[0], edge_ids->shape[0]);
}

CSR::CSR(IdArray indptr, IdArray indices, IdArray edge_ids, bool is_multigraph)
  : indptr_(indptr), indices_(indices), edge_ids_(edge_ids),
    is_multigraph_(is_multigraph) {
  CHECK(IsValidIdArray(indptr));
  CHECK(IsValidIdArray(indices));
  CHECK(IsValidIdArray(edge_ids));
  CHECK_EQ(indices->shape[0], edge_ids->shape[0]);
}

CSR::CSR(IdArray indptr, IdArray indices, IdArray edge_ids,
         const std::string &shared_mem_name): shared_mem_name_(shared_mem_name) {
  CHECK(IsValidIdArray(indptr));
  CHECK(IsValidIdArray(indices));
  CHECK(IsValidIdArray(edge_ids));
  CHECK_EQ(indices->shape[0], edge_ids->shape[0]);
  const int64_t num_verts = indptr->shape[0] - 1;
  const int64_t num_edges = indices->shape[0];
  std::tie(indptr_, indices_, edge_ids_) = MapFromSharedMemory(
      shared_mem_name, num_verts, num_edges, true);
  // copy the given data into the shared memory arrays
  indptr_.CopyFrom(indptr);
  indices_.CopyFrom(indices);
  edge_ids_.CopyFrom(edge_ids);
}

CSR::CSR(IdArray indptr, IdArray indices, IdArray edge_ids, bool is_multigraph,
         const std::string &shared_mem_name): is_multigraph_(is_multigraph),
         shared_mem_name_(shared_mem_name) {
  CHECK(IsValidIdArray(indptr));
  CHECK(IsValidIdArray(indices));
  CHECK(IsValidIdArray(edge_ids));
  CHECK_EQ(indices->shape[0], edge_ids->shape[0]);
  const int64_t num_verts = indptr->shape[0] - 1;
  const int64_t num_edges = indices->shape[0];
  std::tie(indptr_, indices_, edge_ids_) = MapFromSharedMemory(
      shared_mem_name, num_verts, num_edges, true);
  // copy the given data into the shared memory arrays
  indptr_.CopyFrom(indptr);
  indices_.CopyFrom(indices);
  edge_ids_.CopyFrom(edge_ids);
}

CSR::CSR(const std::string &shared_mem_name,
         int64_t num_verts, int64_t num_edges, bool is_multigraph)
  : is_multigraph_(is_multigraph), shared_mem_name_(shared_mem_name) {
  std::tie(indptr_, indices_, edge_ids_) = MapFromSharedMemory(
      shared_mem_name, num_verts, num_edges, false);
}

bool CSR::IsMultigraph() const {
  // The lambda will be called the first time to initialize the is_multigraph flag.
  return const_cast<CSR*>(this)->is_multigraph_.Get([this] () {
      const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
      const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
      for (dgl_id_t src = 0; src < NumVertices(); ++src) {
        std::unordered_set<dgl_id_t> hashmap;
        for (dgl_id_t eid = indptr_data[src]; eid < indptr_data[src+1]; ++eid) {
          const dgl_id_t dst = indices_data[eid];
          if (hashmap.count(dst)) {
            return true;
          } else {
            hashmap.insert(dst);
          }
        }
      }
      return false;
    });
}

CSR::EdgeArray CSR::OutEdges(dgl_id_t vid) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* edge_ids_data = static_cast<dgl_id_t*>(edge_ids_->data);
  const dgl_id_t off = indptr_data[vid];
  const int64_t len = OutDegree(vid);
  IdArray src = NewIdArray(len);
  IdArray dst = NewIdArray(len);
  IdArray eid = NewIdArray(len);
  dgl_id_t* src_data = static_cast<dgl_id_t*>(src->data);
  dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst->data);
  dgl_id_t* eid_data = static_cast<dgl_id_t*>(eid->data);
  std::fill(src_data, src_data + len, vid);
  std::copy(indices_data + off, indices_data + off + len, dst_data);
  std::copy(edge_ids_data + off, edge_ids_data + off + len, eid_data);
  return CSR::EdgeArray{src, dst, eid};
}

CSR::EdgeArray CSR::OutEdges(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* edge_ids_data = static_cast<dgl_id_t*>(edge_ids_->data);
  const auto len = vids->shape[0];
  const dgl_id_t* vid_data = static_cast<dgl_id_t*>(vids->data);
  int64_t rstlen = 0;
  for (int64_t i = 0; i < len; ++i) {
    dgl_id_t vid = vid_data[i];
    CHECK(HasVertex(vid)) << "Invalid vertex: " << vid;
    rstlen += OutDegree(vid);
  }
  IdArray src = NewIdArray(rstlen);
  IdArray dst = NewIdArray(rstlen);
  IdArray eid = NewIdArray(rstlen);
  dgl_id_t* src_ptr = static_cast<dgl_id_t*>(src->data);
  dgl_id_t* dst_ptr = static_cast<dgl_id_t*>(dst->data);
  dgl_id_t* eid_ptr = static_cast<dgl_id_t*>(eid->data);
  for (int64_t i = 0; i < len; ++i) {
    const dgl_id_t vid = vid_data[i];
    const dgl_id_t off = indptr_data[vid];
    const int64_t deg = OutDegree(vid);
    if (deg == 0)
      continue;
    const auto *succ = indices_data + off;
    const auto *eids = edge_ids_data + off;
    for (int64_t j = 0; j < deg; ++j) {
      *(src_ptr++) = vid;
      *(dst_ptr++) = succ[j];
      *(eid_ptr++) = eids[j];
    }
  }
  return CSR::EdgeArray{src, dst, eid};
}

DegreeArray CSR::OutDegrees(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto len = vids->shape[0];
  const dgl_id_t* vid_data = static_cast<dgl_id_t*>(vids->data);
  DegreeArray rst = DegreeArray::Empty({len}, vids->dtype, vids->ctx);
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);
  for (int64_t i = 0; i < len; ++i) {
    const auto vid = vid_data[i];
    CHECK(HasVertex(vid)) << "Invalid vertex: " << vid;
    rst_data[i] = OutDegree(vid);
  }
  return rst;
}

bool CSR::HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const {
  CHECK(HasVertex(src)) << "Invalid vertex id: " << src;
  CHECK(HasVertex(dst)) << "Invalid vertex id: " << dst;
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  for (dgl_id_t i = indptr_data[src]; i < indptr_data[src+1]; ++i) {
    if (indices_data[i] == dst) {
      return true;
    }
  }
  return false;
}

IdArray CSR::Successors(dgl_id_t vid, uint64_t radius) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  CHECK(radius == 1) << "invalid radius: " << radius;
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const int64_t len = indptr_data[vid + 1] - indptr_data[vid];
  IdArray rst = NewIdArray(len);
  dgl_id_t* rst_data = static_cast<dgl_id_t*>(rst->data);
  std::copy(indices_data + indptr_data[vid],
            indices_data + indptr_data[vid + 1],
            rst_data);
  return rst;
}

IdArray CSR::EdgeId(dgl_id_t src, dgl_id_t dst) const {
  // TODO(minjie): use more efficient binary search when the column indices
  //   are also sorted.
  CHECK(HasVertex(src)) << "invalid vertex: " << src;
  CHECK(HasVertex(dst)) << "invalid vertex: " << dst;
  std::vector<dgl_id_t> ids;
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(edge_ids_->data);
  for (dgl_id_t i = indptr_data[src]; i < indptr_data[src+1]; ++i) {
    if (indices_data[i] == dst) {
      ids.push_back(eid_data[i]);
    }
  }
  return VecToIdArray(ids);
}

CSR::EdgeArray CSR::EdgeIds(IdArray src_ids, IdArray dst_ids) const {
  // TODO(minjie): more efficient implementation for simple graph
  CHECK(IsValidIdArray(src_ids)) << "Invalid src id array.";
  CHECK(IsValidIdArray(dst_ids)) << "Invalid dst id array.";
  const auto srclen = src_ids->shape[0];
  const auto dstlen = dst_ids->shape[0];

  CHECK((srclen == dstlen) || (srclen == 1) || (dstlen == 1))
    << "Invalid src and dst id array.";

  const int src_stride = (srclen == 1 && dstlen != 1) ? 0 : 1;
  const int dst_stride = (dstlen == 1 && srclen != 1) ? 0 : 1;
  const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_ids->data);
  const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_ids->data);

  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(edge_ids_->data);

  std::vector<dgl_id_t> src, dst, eid;

  for (int64_t i = 0, j = 0; i < srclen && j < dstlen; i += src_stride, j += dst_stride) {
    const dgl_id_t src_id = src_data[i], dst_id = dst_data[j];
    CHECK(HasVertex(src_id) && HasVertex(dst_id)) <<
        "invalid edge: " << src_id << " -> " << dst_id;
    for (dgl_id_t i = indptr_data[src_id]; i < indptr_data[src_id+1]; ++i) {
      if (indices_data[i] == dst_id) {
          src.push_back(src_id);
          dst.push_back(dst_id);
          eid.push_back(eid_data[i]);
      }
    }
  }
  return CSR::EdgeArray{VecToIdArray(src), VecToIdArray(dst), VecToIdArray(eid)};
}

CSR::EdgeArray CSR::Edges(const std::string &order) const {
  CHECK(order.empty() || order == std::string("srcdst"))
    << "COO only support Edges of order \"srcdst\","
    << " but got \"" << order << "\".";
  const int64_t rstlen = NumEdges();
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  IdArray rst_src = NewIdArray(rstlen);
  dgl_id_t* rst_src_data = static_cast<dgl_id_t*>(rst_src->data);

  // If sorted, the returned edges are sorted by the source Id and dest Id.
  for (dgl_id_t src = 0; src < NumVertices(); ++src) {
    std::fill(rst_src_data + indptr_data[src],
              rst_src_data + indptr_data[src + 1],
              src);
  }

  return CSR::EdgeArray{rst_src, indices_, edge_ids_};
}

Subgraph CSR::VertexSubgraph(IdArray vids) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  IdHashMap hashmap(vids);
  const dgl_id_t* vid_data = static_cast<dgl_id_t*>(vids->data);
  const int64_t len = vids->shape[0];

  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(edge_ids_->data);

  std::vector<dgl_id_t> sub_indptr, sub_indices, sub_eids, induced_edges;
  sub_indptr.resize(len + 1, 0);
  const dgl_id_t kInvalidId = len + 1;
  for (int64_t i = 0; i < len; ++i) {
    // NOTE: newv == i
    const dgl_id_t oldv = vid_data[i];
    CHECK(HasVertex(oldv)) << "Invalid vertex: " << oldv;
    for (dgl_id_t olde = indptr_data[oldv]; olde < indptr_data[oldv+1]; ++olde) {
      const dgl_id_t oldu = indices_data[olde];
      const dgl_id_t newu = hashmap.Map(oldu, kInvalidId);
      if (newu != kInvalidId) {
        ++sub_indptr[i];
        sub_indices.push_back(newu);
        induced_edges.push_back(eid_data[olde]);
      }
    }
  }
  sub_eids.resize(sub_indices.size());
  std::iota(sub_eids.begin(), sub_eids.end(), 0);

  // cumsum sub_indptr
  for (int64_t i = 0, cumsum = 0; i < len; ++i) {
    const dgl_id_t temp = sub_indptr[i];
    sub_indptr[i] = cumsum;
    cumsum += temp;
  }
  sub_indptr[len] = sub_indices.size();

  CSRPtr subcsr(new CSR(
        VecToIdArray(sub_indptr), VecToIdArray(sub_indices), VecToIdArray(sub_eids)));
  return Subgraph{subcsr, vids, VecToIdArray(induced_edges)};
}

// complexity: time O(E + V), space O(1)
CSRPtr CSR::Transpose() const {
  const int64_t N = NumVertices();
  const int64_t M = NumEdges();
  const dgl_id_t* Ap = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* Aj = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* Ax = static_cast<dgl_id_t*>(edge_ids_->data);
  IdArray ret_indptr = NewIdArray(N + 1);
  IdArray ret_indices = NewIdArray(M);
  IdArray ret_edge_ids = NewIdArray(M);
  dgl_id_t* Bp = static_cast<dgl_id_t*>(ret_indptr->data);
  dgl_id_t* Bi = static_cast<dgl_id_t*>(ret_indices->data);
  dgl_id_t* Bx = static_cast<dgl_id_t*>(ret_edge_ids->data);

  std::fill(Bp, Bp + N, 0);

  for (int64_t j = 0; j < M; ++j) {
    Bp[Aj[j]]++;
  }

  // cumsum
  for (int64_t i = 0, cumsum = 0; i < N; ++i) {
    const dgl_id_t temp = Bp[i];
    Bp[i] = cumsum;
    cumsum += temp;
  }
  Bp[N] = M;

  for (int64_t i = 0; i < N; ++i) {
    for (dgl_id_t j = Ap[i]; j < Ap[i+1]; ++j) {
      const dgl_id_t dst = Aj[j];
      Bi[Bp[dst]] = i;
      Bx[Bp[dst]] = Ax[j];
      Bp[dst]++;
    }
  }

  // correct the indptr
  for (int64_t i = 0, last = 0; i <= N; ++i) {
    dgl_id_t temp = Bp[i];
    Bp[i] = last;
    last = temp;
  }

  return CSRPtr(new CSR(ret_indptr, ret_indices, ret_edge_ids));
}

// complexity: time O(E + V), space O(1)
COOPtr CSR::ToCOO() const {
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(indptr_->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(indices_->data);
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(edge_ids_->data);
  IdArray ret_src = NewIdArray(NumEdges());
  IdArray ret_dst = NewIdArray(NumEdges());
  dgl_id_t* ret_src_data = static_cast<dgl_id_t*>(ret_src->data);
  dgl_id_t* ret_dst_data = static_cast<dgl_id_t*>(ret_dst->data);
  // scatter by edge id
  for (dgl_id_t src = 0; src < NumVertices(); ++src) {
    for (dgl_id_t eid = indptr_data[src]; eid < indptr_data[src + 1]; ++eid) {
      const dgl_id_t dst = indices_data[eid];
      ret_src_data[eid_data[eid]] = src;
      ret_dst_data[eid_data[eid]] = dst;
    }
  }
  return COOPtr(new COO(NumVertices(), ret_src, ret_dst));
}

CSR CSR::CopyTo(const DLContext& ctx) const {
  if (Context() == ctx) {
    return *this;
  } else {
    // TODO(minjie): change to use constructor later
    CSR ret;
    ret.indptr_ = indptr_.CopyTo(ctx);
    ret.indices_ = indices_.CopyTo(ctx);
    ret.edge_ids_ = edge_ids_.CopyTo(ctx);
    ret.is_multigraph_ = is_multigraph_;
    return ret;
  }
}

CSR CSR::CopyToSharedMem(const std::string &name) const {
  if (IsSharedMem()) {
    CHECK(name == shared_mem_name_);
    return *this;
  } else {
    return CSR(indptr_, indices_, edge_ids_, name);
  }
}

CSR CSR::AsNumBits(uint8_t bits) const {
  if (NumBits() == bits) {
    return *this;
  } else {
    // TODO(minjie): change to use constructor later
    CSR ret;
    ret.indptr_ = dgl::AsNumBits(indptr_, bits);
    ret.indices_ = dgl::AsNumBits(indices_, bits);
    ret.edge_ids_ = dgl::AsNumBits(edge_ids_, bits);
    ret.is_multigraph_ = is_multigraph_;
    return ret;
  }
}

//////////////////////////////////////////////////////////
//
// COO graph implementation
//
//////////////////////////////////////////////////////////
COO::COO(int64_t num_vertices, IdArray src, IdArray dst)
  : num_vertices_(num_vertices), src_(src), dst_(dst) {
  CHECK(IsValidIdArray(src));
  CHECK(IsValidIdArray(dst));
  CHECK_EQ(src->shape[0], dst->shape[0]);
}

COO::COO(int64_t num_vertices, IdArray src, IdArray dst, bool is_multigraph)
  : num_vertices_(num_vertices), src_(src), dst_(dst), is_multigraph_(is_multigraph) {
  CHECK(IsValidIdArray(src));
  CHECK(IsValidIdArray(dst));
  CHECK_EQ(src->shape[0], dst->shape[0]);
}

bool COO::IsMultigraph() const {
  // The lambda will be called the first time to initialize the is_multigraph flag.
  return const_cast<COO*>(this)->is_multigraph_.Get([this] () {
      std::unordered_set<std::pair<dgl_id_t, dgl_id_t>, PairHash> hashmap;
      const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_->data);
      const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_->data);
      for (dgl_id_t eid = 0; eid < NumEdges(); ++eid) {
        const auto& p = std::make_pair(src_data[eid], dst_data[eid]);
        if (hashmap.count(p)) {
          return true;
        } else {
          hashmap.insert(p);
        }
      }
      return false;
    });
}

COO::EdgeArray COO::FindEdges(IdArray eids) const {
  CHECK(IsValidIdArray(eids)) << "Invalid edge id array";
  dgl_id_t* eid_data = static_cast<dgl_id_t*>(eids->data);
  int64_t len = eids->shape[0];
  IdArray rst_src = NewIdArray(len);
  IdArray rst_dst = NewIdArray(len);
  dgl_id_t* rst_src_data = static_cast<dgl_id_t*>(rst_src->data);
  dgl_id_t* rst_dst_data = static_cast<dgl_id_t*>(rst_dst->data);

  for (int64_t i = 0; i < len; i++) {
    auto edge = COO::FindEdge(eid_data[i]);
    rst_src_data[i] = edge.first;
    rst_dst_data[i] = edge.second;
  }

  return COO::EdgeArray{rst_src, rst_dst, eids};
}

COO::EdgeArray COO::Edges(const std::string &order) const {
  const int64_t rstlen = NumEdges();
  CHECK(order.empty() || order == std::string("eid"))
    << "COO only support Edges of order \"eid\", but got \""
    << order << "\".";
  IdArray rst_eid = NewIdArray(rstlen);
  dgl_id_t* rst_eid_data = static_cast<dgl_id_t*>(rst_eid->data);
  std::iota(rst_eid_data, rst_eid_data + rstlen, 0);
  return EdgeArray{src_, dst_, rst_eid};
}

Subgraph COO::EdgeSubgraph(IdArray eids, bool preserve_nodes) const {
  CHECK(IsValidIdArray(eids)) << "Invalid edge id array.";
  const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_->data);
  const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_->data);
  const dgl_id_t* eids_data = static_cast<dgl_id_t*>(eids->data);
  IdArray new_src = NewIdArray(eids->shape[0]);
  IdArray new_dst = NewIdArray(eids->shape[0]);
  dgl_id_t* new_src_data = static_cast<dgl_id_t*>(new_src->data);
  dgl_id_t* new_dst_data = static_cast<dgl_id_t*>(new_dst->data);
  if (!preserve_nodes) {
    dgl_id_t newid = 0;
    std::unordered_map<dgl_id_t, dgl_id_t> oldv2newv;

    for (int64_t i = 0; i < eids->shape[0]; ++i) {
      const dgl_id_t eid = eids_data[i];
      const dgl_id_t src = src_data[eid];
      const dgl_id_t dst = dst_data[eid];
      if (!oldv2newv.count(src)) {
        oldv2newv[src] = newid++;
      }
      if (!oldv2newv.count(dst)) {
        oldv2newv[dst] = newid++;
      }
      *(new_src_data++) = oldv2newv[src];
      *(new_dst_data++) = oldv2newv[dst];
    }

    // induced nodes
    IdArray induced_nodes = NewIdArray(newid);
    dgl_id_t* induced_nodes_data = static_cast<dgl_id_t*>(induced_nodes->data);
    for (const auto& kv : oldv2newv) {
      induced_nodes_data[kv.second] = kv.first;
    }

    COOPtr subcoo(new COO(newid, new_src, new_dst));
    return Subgraph{subcoo, induced_nodes, eids};
  } else {
    for (int64_t i = 0; i < eids->shape[0]; ++i) {
      const dgl_id_t eid = eids_data[i];
      const dgl_id_t src = src_data[eid];
      const dgl_id_t dst = dst_data[eid];
      *(new_src_data++) = src;
      *(new_dst_data++) = dst;
    }

    IdArray induced_nodes = NewIdArray(NumVertices());
    dgl_id_t* induced_nodes_data = static_cast<dgl_id_t*>(induced_nodes->data);
    for (int64_t i = 0; i < NumVertices(); ++i)
      *(induced_nodes_data++) = i;

    COOPtr subcoo(new COO(NumVertices(), new_src, new_dst));
    return Subgraph{subcoo, induced_nodes, eids};
  }
}

// complexity: time O(E + V), space O(1)
CSRPtr COO::ToCSR() const {
  const int64_t N = num_vertices_;
  const int64_t M = src_->shape[0];
  const dgl_id_t* src_data = static_cast<dgl_id_t*>(src_->data);
  const dgl_id_t* dst_data = static_cast<dgl_id_t*>(dst_->data);
  IdArray indptr = NewIdArray(N + 1);
  IdArray indices = NewIdArray(M);
  IdArray edge_ids = NewIdArray(M);

  dgl_id_t* Bp = static_cast<dgl_id_t*>(indptr->data);
  dgl_id_t* Bi = static_cast<dgl_id_t*>(indices->data);
  dgl_id_t* Bx = static_cast<dgl_id_t*>(edge_ids->data);

  std::fill(Bp, Bp + N, 0);

  for (int64_t i = 0; i < M; ++i) {
    Bp[src_data[i]]++;
  }

  // cumsum
  for (int64_t i = 0, cumsum = 0; i < N; ++i) {
    const dgl_id_t temp = Bp[i];
    Bp[i] = cumsum;
    cumsum += temp;
  }
  Bp[N] = M;

  for (int64_t i = 0; i < M; ++i) {
    const dgl_id_t src = src_data[i];
    const dgl_id_t dst = dst_data[i];
    Bi[Bp[src]] = dst;
    Bx[Bp[src]] = i;
    Bp[src]++;
  }

  // correct the indptr
  for (int64_t i = 0, last = 0; i <= N; ++i) {
    dgl_id_t temp = Bp[i];
    Bp[i] = last;
    last = temp;
  }

  return CSRPtr(new CSR(indptr, indices, edge_ids));
}

COO COO::CopyTo(const DLContext& ctx) const {
  if (Context() == ctx) {
    return *this;
  } else {
    // TODO(minjie): change to use constructor later
    COO ret;
    ret.num_vertices_ = num_vertices_;
    ret.src_ = src_.CopyTo(ctx);
    ret.dst_ = dst_.CopyTo(ctx);
    ret.is_multigraph_ = is_multigraph_;
    return ret;
  }
}

COO COO::CopyToSharedMem(const std::string &name) const {
  LOG(FATAL) << "COO doesn't supprt shared memory yet";
}

COO COO::AsNumBits(uint8_t bits) const {
  if (NumBits() == bits) {
    return *this;
  } else {
    // TODO(minjie): change to use constructor later
    COO ret;
    ret.num_vertices_ = num_vertices_;
    ret.src_ = dgl::AsNumBits(src_, bits);
    ret.dst_ = dgl::AsNumBits(dst_, bits);
    ret.is_multigraph_ = is_multigraph_;
    return ret;
  }
}

//////////////////////////////////////////////////////////
//
// immutable graph implementation
//
//////////////////////////////////////////////////////////

ImmutableGraph::EdgeArray ImmutableGraph::Edges(const std::string &order) const {
  if (order.empty()) {
    // arbitrary order
    if (in_csr_) {
      // transpose
      const auto& edges = in_csr_->Edges(order);
      return EdgeArray{edges.dst, edges.src, edges.id};
    } else {
      return AnyGraph()->Edges(order);
    }
  } else if (order == std::string("srcdst")) {
    // TODO(minjie): CSR only guarantees "src" to be sorted.
    //   Maybe we should relax this requirement?
    return GetOutCSR()->Edges(order);
  } else if (order == std::string("eid")) {
    return GetCOO()->Edges(order);
  } else {
    LOG(FATAL) << "Unsupported order request: " << order;
  }
  return {};
}

Subgraph ImmutableGraph::VertexSubgraph(IdArray vids) const {
  // We prefer to generate a subgraph from out-csr.
  auto sg = GetOutCSR()->VertexSubgraph(vids);
  CSRPtr subcsr = std::dynamic_pointer_cast<CSR>(sg.graph);
  return Subgraph{GraphPtr(new ImmutableGraph(subcsr)),
                  sg.induced_vertices, sg.induced_edges};
}

Subgraph ImmutableGraph::EdgeSubgraph(IdArray eids, bool preserve_nodes) const {
  // We prefer to generate a subgraph from out-csr.
  auto sg = GetCOO()->EdgeSubgraph(eids, preserve_nodes);
  COOPtr subcoo = std::dynamic_pointer_cast<COO>(sg.graph);
  return Subgraph{GraphPtr(new ImmutableGraph(subcoo)),
                  sg.induced_vertices, sg.induced_edges};
}

std::vector<IdArray> ImmutableGraph::GetAdj(bool transpose, const std::string &fmt) const {
  // TODO(minjie): Our current semantics of adjacency matrix is row for dst nodes and col for
  //   src nodes. Therefore, we need to flip the transpose flag. For example, transpose=False
  //   is equal to in edge CSR.
  //   We have this behavior because previously we use framework's SPMM and we don't cache
  //   reverse adj. This is not intuitive and also not consistent with networkx's
  //   to_scipy_sparse_matrix. With the upcoming custom kernel change, we should change the
  //   behavior and make row for src and col for dst.
  if (fmt == std::string("csr")) {
    return transpose? GetOutCSR()->GetAdj(false, "csr") : GetInCSR()->GetAdj(false, "csr");
  } else if (fmt == std::string("coo")) {
    return GetCOO()->GetAdj(!transpose, fmt);
  } else {
    LOG(FATAL) << "unsupported adjacency matrix format: " << fmt;
    return {};
  }
}

ImmutableGraph ImmutableGraph::ToImmutable(const GraphInterface* graph) {
  const ImmutableGraph* ig = dynamic_cast<const ImmutableGraph*>(graph);
  if (ig) {
    return *ig;
  } else {
    const auto& adj = graph->GetAdj(true, "csr");
    CSRPtr csr(new CSR(adj[0], adj[1], adj[2]));
    return ImmutableGraph(nullptr, csr);
  }
}

ImmutableGraph ImmutableGraph::CopyTo(const DLContext& ctx) const {
  if (ctx == Context()) {
    return *this;
  }
  // TODO(minjie): since we don't have GPU implementation of COO<->CSR,
  //   we make sure that this graph (on CPU) has materialized CSR,
  //   and then copy them to other context (usually GPU). This should
  //   be fixed later.
  CSRPtr new_incsr = CSRPtr(new CSR(GetInCSR()->CopyTo(ctx)));
  CSRPtr new_outcsr = CSRPtr(new CSR(GetOutCSR()->CopyTo(ctx)));
  return ImmutableGraph(new_incsr, new_outcsr);
}

ImmutableGraph ImmutableGraph::CopyToSharedMem(const std::string &edge_dir,
                                               const std::string &name) const {
  CSRPtr new_incsr, new_outcsr;
  std::string shared_mem_name = GetSharedMemName(name, edge_dir);
  if (edge_dir == "in")
    new_incsr = CSRPtr(new CSR(GetInCSR()->CopyToSharedMem(shared_mem_name)));
  else if (edge_dir == "out")
    new_outcsr = CSRPtr(new CSR(GetOutCSR()->CopyToSharedMem(shared_mem_name)));
  return ImmutableGraph(new_incsr, new_outcsr, name);
}

ImmutableGraph ImmutableGraph::AsNumBits(uint8_t bits) const {
  if (NumBits() == bits) {
    return *this;
  } else {
    // TODO(minjie): since we don't have int32 operations,
    //   we make sure that this graph (on CPU) has materialized CSR,
    //   and then copy them to other context (usually GPU). This should
    //   be fixed later.
    CSRPtr new_incsr = CSRPtr(new CSR(GetInCSR()->AsNumBits(bits)));
    CSRPtr new_outcsr = CSRPtr(new CSR(GetOutCSR()->AsNumBits(bits)));
    return ImmutableGraph(new_incsr, new_outcsr);
  }
}

}  // namespace dgl
