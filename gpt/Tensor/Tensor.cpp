#include "Tensor.hpp"
#include "TensorCuda.cuh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

Tensor::Tensor() = default;

Tensor::Tensor(const std::vector<size_t> &shape, bool requires_grad)
    : offset_(0), shape_(shape), strides_(contiguous_strides(shape)), requires_grad_(requires_grad)
{
    data_ = std::make_shared<std::vector<double>>(numel(shape), 0.0);
    if (requires_grad_)
    {
        grad_ = std::make_shared<std::vector<double>>(numel(shape), 0.0);
    }
}

Tensor::Tensor(const std::vector<size_t> &shape, double fill_value, bool requires_grad)
    : Tensor(shape, requires_grad)
{
    std::fill(data_->begin(), data_->end(), fill_value);
}

Tensor Tensor::zeros(const std::vector<size_t> &shape, bool requires_grad)
{
    return Tensor(shape, requires_grad);
}

Tensor Tensor::ones(const std::vector<size_t> &shape, bool requires_grad)
{
    return Tensor(shape, 1.0, requires_grad);
}

Tensor Tensor::full(const std::vector<size_t> &shape, double value, bool requires_grad)
{
    return Tensor(shape, value, requires_grad);
}

Tensor Tensor::randomNormal(const std::vector<size_t> &shape, double mean, double stddev, bool requires_grad)
{
    Tensor t(shape, requires_grad);
    t.random(mean, stddev);
    return t;
}

Tensor Tensor::xavier(const std::vector<size_t> &shape, bool requires_grad)
{
    if (shape.size() < 2)
    {
        throw std::invalid_argument("xavier() expects shape [fan_out, fan_in, ...]");
    }
    size_t fan_out = shape[0];
    size_t fan_in = shape[1];
    double limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));

    Tensor t(shape, requires_grad);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (auto &x : *t.data_)
        x = dist(gen);
    return t;
}

Tensor Tensor::he(const std::vector<size_t> &shape, bool requires_grad)
{
    if (shape.size() < 2)
    {
        throw std::invalid_argument("he() expects shape [fan_out, fan_in, ...]");
    }
    size_t fan_in = shape[1];
    double stddev = std::sqrt(2.0 / static_cast<double>(fan_in));

    Tensor t(shape, requires_grad);
    t.random(0.0, stddev);
    return t;
}

std::vector<size_t> Tensor::contiguous_strides(const std::vector<size_t> &shape)
{
    std::vector<size_t> strides(shape.size());
    size_t acc = 1;
    for (size_t i = shape.size(); i-- > 0;)
    {
        strides[i] = acc;
        acc *= shape[i];
    }
    return strides;
}

size_t Tensor::numel(const std::vector<size_t> &shape)
{
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
}

const std::vector<size_t> &Tensor::shape() const { return shape_; }
const std::vector<size_t> &Tensor::strides() const { return strides_; }
size_t Tensor::ndim() const { return shape_.size(); }
size_t Tensor::size() const { return numel(shape_); }

size_t Tensor::dim(size_t axis) const
{
    if (axis >= shape_.size())
        throw std::out_of_range("dim(): axis out of range");
    return shape_[axis];
}

bool Tensor::is_contiguous() const
{
    return strides_ == contiguous_strides(shape_);
}

bool Tensor::requires_grad() const { return requires_grad_; }
void Tensor::set_requires_grad(bool value)
{
    requires_grad_ = value;
    if (requires_grad_)
        ensure_grad();
}
bool Tensor::defined() const { return static_cast<bool>(data_); }

size_t Tensor::rows() const { return shape_.empty() ? 0 : shape_[0]; }
size_t Tensor::cols() const
{
    if (shape_.size() > 1)
        return shape_[1];
    if (shape_.size() == 1)
        return 1;
    return 0;
}

std::vector<double> &Tensor::data()
{
    if (!is_contiguous())
    {
        throw std::runtime_error("data(): tensor is a non-contiguous view; call contiguous() first");
    }
    if (offset_ != 0)
    {
        throw std::runtime_error("data(): tensor has non-zero offset; call contiguous() first");
    }
    return *data_;
}

const std::vector<double> &Tensor::data() const
{
    if (!is_contiguous() || offset_ != 0)
    {
        throw std::runtime_error("data(): tensor is a non-contiguous/offset view; call contiguous() first");
    }
    return *data_;
}

std::vector<double> &Tensor::grad()
{
    const_cast<Tensor *>(this)->ensure_grad();
    if (!is_contiguous() || offset_ != 0)
    {
        throw std::runtime_error("grad(): tensor is a non-contiguous/offset view");
    }
    return *grad_;
}

const std::vector<double> &Tensor::grad() const
{
    if (!grad_)
        throw std::runtime_error("grad(): gradient buffer not allocated (requires_grad is false)");
    if (!is_contiguous() || offset_ != 0)
    {
        throw std::runtime_error("grad(): tensor is a non-contiguous/offset view");
    }
    return *grad_;
}

size_t Tensor::flat_offset(const std::vector<size_t> &index) const
{
    if (index.size() != shape_.size())
    {
        throw std::invalid_argument("index rank mismatch");
    }
    size_t off = offset_;
    for (size_t i = 0; i < index.size(); ++i)
    {
        if (index[i] >= shape_[i])
            throw std::out_of_range("index out of bounds");
        off += index[i] * strides_[i];
    }
    return off;
}

double &Tensor::at(const std::vector<size_t> &index) { return (*data_)[flat_offset(index)]; }
const double &Tensor::at(const std::vector<size_t> &index) const { return (*data_)[flat_offset(index)]; }

double &Tensor::operator()(std::initializer_list<size_t> index) { return at(std::vector<size_t>(index)); }
const double &Tensor::operator()(std::initializer_list<size_t> index) const { return at(std::vector<size_t>(index)); }

double &Tensor::operator()(size_t i)
{
    if (is_contiguous())
        return (*data_)[offset_ + i];
    std::vector<size_t> idx(shape_.size());
    size_t rem = i;
    for (size_t d = shape_.size(); d-- > 0;)
    {
        idx[d] = rem % shape_[d];
        rem /= shape_[d];
    }
    return at(idx);
}

const double &Tensor::operator()(size_t i) const
{
    if (is_contiguous())
        return (*data_)[offset_ + i];
    std::vector<size_t> idx(shape_.size());
    size_t rem = i;
    for (size_t d = shape_.size(); d-- > 0;)
    {
        idx[d] = rem % shape_[d];
        rem /= shape_[d];
    }
    return at(idx);
}

double &Tensor::operator()(size_t row, size_t col)
{
    if (shape_.size() < 2)
        throw std::runtime_error("2D access used on a rank<2 tensor");
    return at({row, col});
}

const double &Tensor::operator()(size_t row, size_t col) const
{
    if (shape_.size() < 2)
        throw std::runtime_error("2D access used on a rank<2 tensor");
    return at({row, col});
}

void Tensor::ensure_grad()
{
    if (!grad_)
    {
        grad_ = std::make_shared<std::vector<double>>(data_->size(), 0.0);
    }
}

void Tensor::zero_grad()
{
    ensure_grad();
    std::fill(grad_->begin(), grad_->end(), 0.0);
}

void Tensor::zero() { fill(0.0); }

void Tensor::fill(double value)
{
    for (size_t i = 0; i < size(); ++i)
        (*this)(i) = value;
}

void Tensor::random(double mean, double stddev)
{
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(mean, stddev);
    for (size_t i = 0; i < size(); ++i)
        (*this)(i) = dist(gen);
}

Tensor Tensor::reshape(const std::vector<size_t> &new_shape) const
{
    if (numel(new_shape) != size())
    {
        throw std::invalid_argument("reshape(): element count mismatch");
    }
    if (is_contiguous())
    {
        Tensor out = *this;
        out.shape_ = new_shape;
        out.strides_ = contiguous_strides(new_shape);
        return out;
    }
    return contiguous().reshape(new_shape);
}

Tensor Tensor::view(const std::vector<size_t> &new_shape) const
{
    if (!is_contiguous())
    {
        throw std::runtime_error("view(): tensor is not contiguous; use reshape() instead");
    }
    return reshape(new_shape);
}

Tensor Tensor::flatten() const
{
    if (shape_.empty())
        return reshape({size()});
    return flatten(0, shape_.size() - 1);
}

Tensor Tensor::flatten(size_t startDim, size_t endDim) const
{
    if (shape_.empty())
        throw std::runtime_error("flatten(): cannot flatten a rank-0 tensor with a dimension range");
    if (endDim >= shape_.size() || startDim > endDim)
        throw std::out_of_range("flatten(): invalid dimension range");

    if (startDim == endDim)
        return *this;

    // Dims [startDim, endDim] can be merged into a single dimension without
    // copying only if they're "stride-contiguous" among themselves, i.e.
    // strides_[i] == strides_[i+1] * shape_[i+1] for each adjacent pair.
    // This holds for a fully contiguous tensor, but also for non-contiguous
    // views where only the merged range happens to be contiguous (e.g. the
    // trailing dims of a permuted tensor).
    bool mergeable = true;
    for (size_t i = startDim; i < endDim; ++i)
    {
        if (strides_[i] != strides_[i + 1] * shape_[i + 1])
        {
            mergeable = false;
            break;
        }
    }

    if (!mergeable)
        return contiguous().flatten(startDim, endDim);

    size_t merged_size = 1;
    for (size_t i = startDim; i <= endDim; ++i)
        merged_size *= shape_[i];

    std::vector<size_t> out_shape;
    std::vector<size_t> out_strides;
    out_shape.reserve(shape_.size() - (endDim - startDim));
    out_strides.reserve(out_shape.capacity());

    for (size_t i = 0; i < startDim; ++i)
    {
        out_shape.push_back(shape_[i]);
        out_strides.push_back(strides_[i]);
    }
    out_shape.push_back(merged_size);
    out_strides.push_back(strides_[endDim]);
    for (size_t i = endDim + 1; i < shape_.size(); ++i)
    {
        out_shape.push_back(shape_[i]);
        out_strides.push_back(strides_[i]);
    }

    Tensor out = *this;
    out.shape_ = std::move(out_shape);
    out.strides_ = std::move(out_strides);
    return out;
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const
{
    if (dim0 >= shape_.size() || dim1 >= shape_.size())
    {
        throw std::out_of_range("transpose(): axis out of range");
    }
    Tensor out = *this;
    std::swap(out.shape_[dim0], out.shape_[dim1]);
    std::swap(out.strides_[dim0], out.strides_[dim1]);
    return out;
}

Tensor Tensor::permute(const std::vector<size_t> &dims) const
{
    if (dims.size() != shape_.size())
    {
        throw std::invalid_argument("permute(): dims size must equal tensor rank");
    }
    Tensor out = *this;
    for (size_t i = 0; i < dims.size(); ++i)
    {
        out.shape_[i] = shape_[dims[i]];
        out.strides_[i] = strides_[dims[i]];
    }
    return out;
}

Tensor Tensor::slice(size_t axis, size_t start, size_t end) const
{
    if (axis >= shape_.size())
        throw std::out_of_range("slice(): axis out of range");
    if (start >= end || end > shape_[axis])
        throw std::out_of_range("slice(): invalid range");

    Tensor out = *this;
    out.offset_ = offset_ + start * strides_[axis];
    out.shape_[axis] = end - start;
    return out;
}

Tensor Tensor::clone() const
{
    Tensor out(shape_, requires_grad_);
    for (size_t i = 0; i < size(); ++i)
        out(i) = (*this)(i);
    return out;
}

Tensor Tensor::contiguous() const
{
    if (is_contiguous() && offset_ == 0)
        return *this;
    return clone();
}

void Tensor::broadcast_shapes(const std::vector<size_t> &a, const std::vector<size_t> &b, std::vector<size_t> &out_shape)
{
    size_t rank = std::max(a.size(), b.size());
    out_shape.assign(rank, 1);
    for (size_t i = 0; i < rank; ++i)
    {
        size_t da = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
        size_t db = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
        if (da != db && da != 1 && db != 1)
        {
            throw std::invalid_argument("shapes are not broadcastable");
        }
        out_shape[i] = std::max(da, db);
    }
}

size_t Tensor::broadcast_offset(const std::vector<size_t> &multi_index, const std::vector<size_t> &out_shape, const std::vector<size_t> &in_shape, const std::vector<size_t> &in_strides, size_t in_offset)
{
    size_t rank = out_shape.size();
    size_t pad = rank - in_shape.size();
    size_t off = in_offset;
    for (size_t i = 0; i < in_shape.size(); ++i)
    {
        size_t coord = multi_index[i + pad];
        size_t dim_size = in_shape[i];
        size_t effective_coord = (dim_size == 1) ? 0 : coord;
        off += effective_coord * in_strides[i];
    }
    return off;
}

Tensor Tensor::add(const Tensor &a, const Tensor &b)
{
    std::vector<size_t> out_shape;
    broadcast_shapes(a.shape_, b.shape_, out_shape);
    Tensor out(out_shape);

    size_t total = out.size();
    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t linear = 0; linear < total; ++linear)
    {
        size_t rem = linear;
        for (size_t d = out_shape.size(); d-- > 0;)
        {
            idx[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        size_t off_a = broadcast_offset(idx, out_shape, a.shape_, a.strides_, a.offset_);
        size_t off_b = broadcast_offset(idx, out_shape, b.shape_, b.strides_, b.offset_);
        out(linear) = (*a.data_)[off_a] + (*b.data_)[off_b];
    }
    return out;
}

Tensor Tensor::subtract(const Tensor &a, const Tensor &b)
{
    std::vector<size_t> out_shape;
    broadcast_shapes(a.shape_, b.shape_, out_shape);
    Tensor out(out_shape);

    size_t total = out.size();
    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t linear = 0; linear < total; ++linear)
    {
        size_t rem = linear;
        for (size_t d = out_shape.size(); d-- > 0;)
        {
            idx[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        size_t off_a = broadcast_offset(idx, out_shape, a.shape_, a.strides_, a.offset_);
        size_t off_b = broadcast_offset(idx, out_shape, b.shape_, b.strides_, b.offset_);
        out(linear) = (*a.data_)[off_a] - (*b.data_)[off_b];
    }
    return out;
}

Tensor Tensor::multiply(const Tensor &a, const Tensor &b)
{
    std::vector<size_t> out_shape;
    broadcast_shapes(a.shape_, b.shape_, out_shape);
    Tensor out(out_shape);

    size_t total = out.size();
    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t linear = 0; linear < total; ++linear)
    {
        size_t rem = linear;
        for (size_t d = out_shape.size(); d-- > 0;)
        {
            idx[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        size_t off_a = broadcast_offset(idx, out_shape, a.shape_, a.strides_, a.offset_);
        size_t off_b = broadcast_offset(idx, out_shape, b.shape_, b.strides_, b.offset_);
        out(linear) = (*a.data_)[off_a] * (*b.data_)[off_b];
    }
    return out;
}

Tensor Tensor::multiply(const Tensor &a, double scalar)
{
    Tensor out(a.shape_);
    for (size_t i = 0; i < a.size(); ++i)
        out(i) = a(i) * scalar;
    return out;
}

Tensor Tensor::divide(const Tensor &a, const Tensor &b)
{
    std::vector<size_t> out_shape;
    broadcast_shapes(a.shape_, b.shape_, out_shape);
    Tensor out(out_shape);

    size_t total = out.size();
    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t linear = 0; linear < total; ++linear)
    {
        size_t rem = linear;
        for (size_t d = out_shape.size(); d-- > 0;)
        {
            idx[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        size_t off_a = broadcast_offset(idx, out_shape, a.shape_, a.strides_, a.offset_);
        size_t off_b = broadcast_offset(idx, out_shape, b.shape_, b.strides_, b.offset_);
        double denom = (*b.data_)[off_b];
        if (denom == 0.0)
            throw std::runtime_error("divide(): division by zero");
        out(linear) = (*a.data_)[off_a] / denom;
    }
    return out;
}

Tensor Tensor::matmul(const Tensor &a, const Tensor &b)
{
    if (a.ndim() < 2 || b.ndim() < 2)
    {
        throw std::invalid_argument("matmul(): both operands must have rank >= 2");
    }
    size_t M = a.shape_[a.ndim() - 2];
    size_t K = a.shape_[a.ndim() - 1];
    size_t K2 = b.shape_[b.ndim() - 2];
    size_t N = b.shape_[b.ndim() - 1];
    if (K != K2)
        throw std::runtime_error("matmul(): inner dimensions mismatch");

    std::vector<size_t> batch_a(a.shape_.begin(), a.shape_.end() - 2);
    std::vector<size_t> batch_b(b.shape_.begin(), b.shape_.end() - 2);
    std::vector<size_t> batch_shape;
    broadcast_shapes(batch_a, batch_b, batch_shape);

    std::vector<size_t> out_shape = batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);
    Tensor out(out_shape);

    size_t batch_count = numel(batch_shape);
    std::vector<size_t> bidx(batch_shape.size(), 0);

    // Gather each batch's A/B matrices into flat, contiguous [batch,M,K] /
    // [batch,K,N] buffers. This costs O(batch*(M*K + K*N)), always cheaper
    // than the O(batch*M*K*N) matmul itself, and turns the general
    // strided/broadcasting case into a plain dense batched GEMM that the
    // GPU path (and the CPU fallback below) can both consume directly.
    std::vector<double> A_flat(batch_count * M * K);
    std::vector<double> B_flat(batch_count * K * N);

    size_t a_row_stride = a.strides_[a.ndim() - 2];
    size_t a_col_stride = a.strides_[a.ndim() - 1];
    size_t b_row_stride = b.strides_[b.ndim() - 2];
    size_t b_col_stride = b.strides_[b.ndim() - 1];

    for (size_t bcount = 0; bcount < batch_count; ++bcount)
    {
        size_t rem = bcount;
        for (size_t d = batch_shape.size(); d-- > 0;)
        {
            bidx[d] = rem % batch_shape[d];
            rem /= batch_shape[d];
        }
        size_t a_batch_off = broadcast_offset(bidx, batch_shape, batch_a, a.strides_, a.offset_);
        size_t b_batch_off = broadcast_offset(bidx, batch_shape, batch_b, b.strides_, b.offset_);

        double *A_dst = A_flat.data() + bcount * M * K;
        for (size_t i = 0; i < M; ++i)
            for (size_t k = 0; k < K; ++k)
                A_dst[i * K + k] = (*a.data_)[a_batch_off + i * a_row_stride + k * a_col_stride];

        double *B_dst = B_flat.data() + bcount * K * N;
        for (size_t k = 0; k < K; ++k)
            for (size_t j = 0; j < N; ++j)
                B_dst[k * N + j] = (*b.data_)[b_batch_off + k * b_row_stride + j * b_col_stride];
    }

#ifdef USE_CUDA
    // GPU launch/copy overhead only pays off once there's real work to do;
    // tiny matmuls (a handful of tokens/heads) are faster on the CPU.
    constexpr size_t kCudaMatmulThreshold = 1u << 16;
    if (batch_count * M * K * N >= kCudaMatmulThreshold)
    {
        cuda_batched_matmul(A_flat.data(), B_flat.data(), out.data().data(),
                             batch_count, M, K, N);
        return out;
    }
#endif

    std::vector<double> &out_data = out.data();
    for (size_t bcount = 0; bcount < batch_count; ++bcount)
    {
        const double *A_src = A_flat.data() + bcount * M * K;
        const double *B_src = B_flat.data() + bcount * K * N;
        double *C_dst = out_data.data() + bcount * M * N;
        for (size_t i = 0; i < M; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                double sum = 0.0;
                for (size_t k = 0; k < K; ++k)
                    sum += A_src[i * K + k] * B_src[k * N + j];
                C_dst[i * N + j] = sum;
            }
        }
    }
    return out;
}

std::vector<Tensor> Tensor::split(const Tensor &tensor, size_t axis, size_t split_size)
{
    if (axis >= tensor.ndim())
        throw std::out_of_range("split(): axis out of range");
    if (split_size == 0)
        throw std::invalid_argument("split(): split_size must be > 0");

    std::vector<Tensor> result;
    size_t dim_size = tensor.shape_[axis];
    for (size_t start = 0; start < dim_size; start += split_size)
    {
        size_t end = std::min(start + split_size, dim_size);
        result.push_back(tensor.slice(axis, start, end));
    }
    return result;
}

Tensor Tensor::concatenate(const std::vector<Tensor> &tensors, size_t axis)
{
    if (tensors.empty())
        throw std::invalid_argument("concatenate(): empty input");

    const std::vector<size_t> &baseShape = tensors[0].shape_;
    if (axis >= baseShape.size())
        throw std::out_of_range("concatenate(): axis out of range");

    std::vector<size_t> out_shape = baseShape;
    out_shape[axis] = 0;

    for (const Tensor &t : tensors)
    {
        if (t.ndim() != baseShape.size())
            throw std::invalid_argument("concatenate(): rank mismatch");
        for (size_t i = 0; i < baseShape.size(); ++i)
        {
            if (i == axis)
                continue;
            if (t.shape_[i] != baseShape[i])
                throw std::invalid_argument("concatenate(): incompatible shapes");
        }
        out_shape[axis] += t.shape_[axis];
    }

    Tensor out(out_shape);

    size_t axis_offset = 0;
    for (const Tensor &t : tensors)
    {
        size_t total = t.size();
        std::vector<size_t> idx(t.shape_.size(), 0);
        for (size_t linear = 0; linear < total; ++linear)
        {
            size_t rem = linear;
            for (size_t d = t.shape_.size(); d-- > 0;)
            {
                idx[d] = rem % t.shape_[d];
                rem /= t.shape_[d];
            }
            std::vector<size_t> out_idx = idx;
            out_idx[axis] += axis_offset;
            out.at(out_idx) = t.at(idx);
        }
        axis_offset += t.shape_[axis];
    }

    return out;
}

Tensor Tensor::stack(const std::vector<Tensor> &tensors, size_t axis)
{
    if (tensors.empty())
        throw std::invalid_argument("stack(): empty input");

    const std::vector<size_t> &baseShape = tensors[0].shape_;
    if (axis > baseShape.size())
        throw std::out_of_range("stack(): axis out of range");

    for (const Tensor &t : tensors)
    {
        if (t.shape_ != baseShape)
            throw std::invalid_argument("stack(): all tensors must have the same shape");
    }

    std::vector<size_t> out_shape;
    out_shape.reserve(baseShape.size() + 1);
    out_shape.insert(out_shape.end(), baseShape.begin(), baseShape.begin() + axis);
    out_shape.push_back(tensors.size());
    out_shape.insert(out_shape.end(), baseShape.begin() + axis, baseShape.end());

    Tensor out(out_shape);

    for (size_t n = 0; n < tensors.size(); ++n)
    {
        const Tensor &t = tensors[n];
        size_t total = t.size();
        std::vector<size_t> idx(baseShape.size(), 0);
        for (size_t linear = 0; linear < total; ++linear)
        {
            size_t rem = linear;
            for (size_t d = baseShape.size(); d-- > 0;)
            {
                idx[d] = rem % baseShape[d];
                rem /= baseShape[d];
            }
            std::vector<size_t> out_idx;
            out_idx.reserve(out_shape.size());
            out_idx.insert(out_idx.end(), idx.begin(), idx.begin() + axis);
            out_idx.push_back(n);
            out_idx.insert(out_idx.end(), idx.begin() + axis, idx.end());
            out.at(out_idx) = t.at(idx);
        }
    }

    return out;
}

Tensor Tensor::sum(std::optional<size_t> axis) const
{
    if (!axis.has_value())
    {
        double total = 0.0;
        for (size_t i = 0; i < size(); ++i)
            total += (*this)(i);
        Tensor out({1});
        out(0) = total;
        return out;
    }

    size_t ax = axis.value();
    if (ax >= shape_.size())
        throw std::out_of_range("sum(): axis out of range");

    std::vector<size_t> out_shape = shape_;
    out_shape[ax] = 1;
    Tensor out(out_shape);

    size_t total = size();
    std::vector<size_t> idx(shape_.size(), 0);
    for (size_t linear = 0; linear < total; ++linear)
    {
        size_t rem = linear;
        for (size_t d = shape_.size(); d-- > 0;)
        {
            idx[d] = rem % shape_[d];
            rem /= shape_[d];
        }
        std::vector<size_t> out_idx = idx;
        out_idx[ax] = 0;
        out.at(out_idx) += at(idx);
    }
    return out;
}

Tensor Tensor::mean(std::optional<size_t> axis) const
{
    Tensor s = sum(axis);
    double denom = axis.has_value() ? static_cast<double>(shape_[axis.value()]) : static_cast<double>(size());
    return multiply(s, 1.0 / denom);
}

Tensor Tensor::variance(std::optional<size_t> axis) const
{
    Tensor m = mean(axis);
    // Broadcast-subtract, square, then mean again.
    Tensor diff = subtract(*this, m);
    Tensor sq = multiply(diff, diff);
    return sq.mean(axis);
}

Tensor Tensor::max(std::optional<size_t> axis) const
{
    if (!axis.has_value())
    {
        double best = (*this)(0);
        for (size_t i = 1; i < size(); ++i)
            best = std::max(best, (*this)(i));
        Tensor out({1});
        out(0) = best;
        return out;
    }

    size_t ax = axis.value();
    if (ax >= shape_.size())
        throw std::out_of_range("max(): axis out of range");

    std::vector<size_t> out_shape = shape_;
    out_shape[ax] = 1;
    Tensor out(out_shape, -std::numeric_limits<double>::infinity());

    size_t total = size();
    std::vector<size_t> idx(shape_.size(), 0);
    for (size_t linear = 0; linear < total; ++linear)
    {
        size_t rem = linear;
        for (size_t d = shape_.size(); d-- > 0;)
        {
            idx[d] = rem % shape_[d];
            rem /= shape_[d];
        }
        std::vector<size_t> out_idx = idx;
        out_idx[ax] = 0;
        double v = at(idx);
        if (v > out.at(out_idx))
            out.at(out_idx) = v;
    }
    return out;
}

Tensor Tensor::argmax(std::optional<size_t> axis) const
{
    if (!axis.has_value())
    {
        double best = (*this)(0);
        size_t best_idx = 0;
        for (size_t i = 1; i < size(); ++i)
        {
            if ((*this)(i) > best)
            {
                best = (*this)(i);
                best_idx = i;
            }
        }
        Tensor out({1});
        out(0) = static_cast<double>(best_idx);
        return out;
    }

    size_t ax = axis.value();
    if (ax >= shape_.size())
        throw std::out_of_range("argmax(): axis out of range");

    std::vector<size_t> out_shape = shape_;
    out_shape[ax] = 1;
    Tensor best_val(out_shape, -std::numeric_limits<double>::infinity());
    Tensor out(out_shape);

    size_t total = size();
    std::vector<size_t> idx(shape_.size(), 0);
    for (size_t linear = 0; linear < total; ++linear)
    {
        size_t rem = linear;
        for (size_t d = shape_.size(); d-- > 0;)
        {
            idx[d] = rem % shape_[d];
            rem /= shape_[d];
        }
        std::vector<size_t> out_idx = idx;
        out_idx[ax] = 0;
        double v = at(idx);
        if (v > best_val.at(out_idx))
        {
            best_val.at(out_idx) = v;
            out.at(out_idx) = static_cast<double>(idx[ax]);
        }
    }
    return out;
}

void Tensor::print() const
{
    if (shape_.size() == 1)
    {
        for (size_t i = 0; i < shape_[0]; ++i)
            std::cout << std::setw(10) << (*this)(i) << " ";
        std::cout << std::endl;
        return;
    }
    if (shape_.size() == 2)
    {
        for (size_t r = 0; r < rows(); ++r)
        {
            for (size_t c = 0; c < cols(); ++c)
                std::cout << std::setw(10) << (*this)(r, c) << " ";
            std::cout << std::endl;
        }
        return;
    }
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i)
        std::cout << shape_[i] << (i + 1 < shape_.size() ? "," : "");
    std::cout << "], " << size() << " elements)" << std::endl;
}

Tensor &Tensor::operator/=(double scalar)
{
    if (scalar == 0.0)
    {
        throw std::runtime_error("Division by zero.");
    }

    for (size_t i = 0; i < data_->size(); ++i)
    {
        (*data_)[i] /= scalar;
    }

    return *this;
}