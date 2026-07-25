#pragma once
#include <vector>
#include <memory>
#include <optional>
#include <initializer_list>
#include <string>

class Tensor
{
public:
    Tensor();
    explicit Tensor(const std::vector<size_t>& shape, bool requires_grad = false);
    Tensor(const std::vector<size_t>& shape, double fill_value, bool requires_grad = false);

    static Tensor zeros(const std::vector<size_t>& shape, bool requires_grad = false);
    static Tensor ones(const std::vector<size_t>& shape, bool requires_grad = false);
    static Tensor full(const std::vector<size_t>& shape, double value, bool requires_grad = false);
    static Tensor randomNormal(const std::vector<size_t>& shape, double mean = 0.0, double stddev = 0.02, bool requires_grad = false);
    static Tensor xavier(const std::vector<size_t>& shape, bool requires_grad = false);
    static Tensor he(const std::vector<size_t>& shape, bool requires_grad = false);

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;
    size_t ndim() const;
    size_t size() const;
    size_t dim(size_t axis) const;
    bool is_contiguous() const;
    bool requires_grad() const;
    void set_requires_grad(bool value);
    bool defined() const;

    std::vector<double>& data();
    const std::vector<double>& data() const;
    std::vector<double>& grad();
    const std::vector<double>& grad() const;

    double& at(const std::vector<size_t>& index);
    const double& at(const std::vector<size_t>& index) const;
    double& operator()(std::initializer_list<size_t> index);
    const double& operator()(std::initializer_list<size_t> index) const;

    double& operator()(size_t i);
    const double& operator()(size_t i) const;

    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    size_t rows() const;
    size_t cols() const;

    void ensure_grad();
    void zero_grad();
    void zero();
    void fill(double value);
    void random(double mean = 0.0, double stddev = 0.02);

    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor view(const std::vector<size_t>& new_shape) const; 
    Tensor flatten() const;
    Tensor transpose(size_t dim0 = 0, size_t dim1 = 1) const;
    Tensor permute(const std::vector<size_t>& dims) const;
    Tensor slice(size_t axis, size_t start, size_t end) const;

    Tensor clone() const;
    Tensor contiguous() const;

    static Tensor add(const Tensor& a, const Tensor& b);
    static Tensor subtract(const Tensor& a, const Tensor& b);
    static Tensor multiply(const Tensor& a, const Tensor& b); 
    static Tensor multiply(const Tensor& a, double scalar);
    static Tensor divide(const Tensor& a, const Tensor& b);
    static Tensor matmul(const Tensor& a, const Tensor& b);

    Tensor sum(std::optional<size_t> axis = std::nullopt) const;
    Tensor mean(std::optional<size_t> axis = std::nullopt) const;
    Tensor variance(std::optional<size_t> axis = std::nullopt) const;
    Tensor max(std::optional<size_t> axis = std::nullopt) const;
    Tensor argmax(std::optional<size_t> axis = std::nullopt) const; 

    void print() const;

private:
    std::shared_ptr<std::vector<double>> data_;
    std::shared_ptr<std::vector<double>> grad_;
    size_t offset_ = 0;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    bool requires_grad_ = false;

    static std::vector<size_t> contiguous_strides(const std::vector<size_t>& shape);
    static size_t numel(const std::vector<size_t>& shape);
    size_t flat_offset(const std::vector<size_t>& index) const;
    static void broadcast_shapes(const std::vector<size_t>& a, const std::vector<size_t>& b, std::vector<size_t>& out_shape);
    static size_t broadcast_offset(const std::vector<size_t>& multi_index, const std::vector<size_t>& out_shape, const std::vector<size_t>& in_shape, const std::vector<size_t>& in_strides, size_t in_offset);
};
