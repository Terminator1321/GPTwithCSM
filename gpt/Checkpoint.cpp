#include "Checkpoint.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

namespace
{
    constexpr char kMagic[4] = {'C', 'S', 'M', '1'};
    constexpr uint32_t kVersion = 1;

    template <typename T>
    void writeRaw(std::ofstream &out, const T &value)
    {
        out.write(reinterpret_cast<const char *>(&value), sizeof(T));
    }

    template <typename T>
    bool readRaw(std::ifstream &in, T &value)
    {
        in.read(reinterpret_cast<char *>(&value), sizeof(T));
        return static_cast<bool>(in);
    }

    void writeDoubles(std::ofstream &out, const std::vector<double> &values)
    {
        if (!values.empty())
        {
            out.write(reinterpret_cast<const char *>(values.data()),
                      static_cast<std::streamsize>(values.size() * sizeof(double)));
        }
    }

    bool readDoubles(std::ifstream &in, std::vector<double> &values, size_t count)
    {
        if (values.size() != count)
        {
            return false;
        }
        if (count > 0)
        {
            in.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(double)));
        }
        return static_cast<bool>(in);
    }

}

bool saveCheckpoint(const std::string &path, GPT &gpt, size_t globalStep, size_t vocabSize, size_t embedDim, size_t maxSeqLen, size_t numHeads, size_t numLayers)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open())
    {
        std::cerr << "Checkpoint: could not open '" << path << "' for writing." << std::endl;
        return false;
    }

    out.write(kMagic, sizeof(kMagic));
    writeRaw(out, kVersion);
    writeRaw(out, static_cast<uint64_t>(vocabSize));
    writeRaw(out, static_cast<uint64_t>(embedDim));
    writeRaw(out, static_cast<uint64_t>(maxSeqLen));
    writeRaw(out, static_cast<uint64_t>(numHeads));
    writeRaw(out, static_cast<uint64_t>(numLayers));
    writeRaw(out, static_cast<uint64_t>(globalStep));

    std::vector<Parameter *> params = gpt.parameters();
    writeRaw(out, static_cast<uint64_t>(params.size()));

    for (Parameter *param : params)
    {
        const std::vector<size_t> &shape = param->shape();
        writeRaw(out, static_cast<uint64_t>(shape.size()));
        for (size_t dim : shape)
        {
            writeRaw(out, static_cast<uint64_t>(dim));
        }

        writeRaw(out, static_cast<uint64_t>(param->value.size()));
        writeDoubles(out, param->value.data());
        writeDoubles(out, param->m.data());
        writeDoubles(out, param->v.data());
    }

    if (!out)
    {
        std::cerr << "Checkpoint: error occurred while writing '" << path << "'." << std::endl;
        return false;
    }

    return true;
}

bool loadCheckpoint(const std::string &path, GPT &gpt, size_t &globalStep, size_t vocabSize, size_t embedDim, size_t maxSeqLen, size_t numHeads, size_t numLayers)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
    {
        return false;
    }

    char magic[4] = {0, 0, 0, 0};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
    {
        std::cerr << "Checkpoint: '" << path << "' is not a valid .csm file." << std::endl;
        return false;
    }

    uint32_t version = 0;
    if (!readRaw(in, version) || version != kVersion)
    {
        std::cerr << "Checkpoint: unsupported .csm version in '" << path << "'." << std::endl;
        return false;
    }

    uint64_t fileVocab = 0, fileEmbed = 0, fileMaxSeq = 0, fileHeads = 0, fileLayers = 0, fileStep = 0;
    if (!readRaw(in, fileVocab) || !readRaw(in, fileEmbed) || !readRaw(in, fileMaxSeq) ||
        !readRaw(in, fileHeads) || !readRaw(in, fileLayers) || !readRaw(in, fileStep))
    {
        std::cerr << "Checkpoint: '" << path << "' is truncated (header)." << std::endl;
        return false;
    }

    if (fileVocab != vocabSize || fileEmbed != embedDim || fileMaxSeq != maxSeqLen ||
        fileHeads != numHeads || fileLayers != numLayers)
    {
        std::cerr << "Checkpoint: architecture in '" << path << "' (vocab=" << fileVocab << ", embed=" << fileEmbed << ", maxSeq=" << fileMaxSeq << ", heads=" << fileHeads << ", layers=" << fileLayers << ") does not match the current model (vocab=" << vocabSize << ", embed=" << embedDim << ", maxSeq=" << maxSeqLen << ", heads=" << numHeads << ", layers=" << numLayers << ")." << std::endl;
        return false;
    }

    uint64_t paramCount = 0;
    if (!readRaw(in, paramCount))
    {
        std::cerr << "Checkpoint: '" << path << "' is truncated (param count)." << std::endl;
        return false;
    }

    std::vector<Parameter *> params = gpt.parameters();
    if (paramCount != params.size())
    {
        std::cerr << "Checkpoint: parameter count mismatch while loading '" << path << "' (file has " << paramCount << ", model has " << params.size() << ")." << std::endl;
        return false;
    }

    for (Parameter *param : params)
    {
        uint64_t ndim = 0;
        if (!readRaw(in, ndim))
        {
            std::cerr << "Checkpoint: '" << path << "' is truncated (shape rank)." << std::endl;
            return false;
        }

        std::vector<size_t> shape(static_cast<size_t>(ndim));
        for (uint64_t d = 0; d < ndim; ++d)
        {
            uint64_t dim = 0;
            if (!readRaw(in, dim))
            {
                std::cerr << "Checkpoint: '" << path << "' is truncated (shape dims)." << std::endl;
                return false;
            }
            shape[static_cast<size_t>(d)] = static_cast<size_t>(dim);
        }

        if (shape != param->shape())
        {
            std::cerr << "Checkpoint: parameter shape mismatch while loading '" << path << "'." << std::endl;
            return false;
        }

        uint64_t count = 0;
        if (!readRaw(in, count))
        {
            std::cerr << "Checkpoint: '" << path << "' is truncated (element count)." << std::endl;
            return false;
        }

        if (static_cast<size_t>(count) != param->value.size())
        {
            std::cerr << "Checkpoint: parameter size mismatch while loading '" << path << "'." << std::endl;
            return false;
        }

        if (!readDoubles(in, param->value.data(), static_cast<size_t>(count)) ||
            !readDoubles(in, param->m.data(), static_cast<size_t>(count)) ||
            !readDoubles(in, param->v.data(), static_cast<size_t>(count)))
        {
            std::cerr << "Checkpoint: '" << path << "' is truncated (parameter data)." << std::endl;
            return false;
        }
    }

    globalStep = static_cast<size_t>(fileStep);
    return true;
}
