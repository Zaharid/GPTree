#include "tree.hpp"

#include "cxxopts.hpp"

#include <exception>
#include <iostream>
#include <istream>
#include <memory>
#include <string>
#include <vector>

const char delimiter = '\t';

using ZKDTree::Data2D;

struct fileinfo {
  Data2D data;
  std::vector<double> y;
};

struct fileinfo parse_csv(std::istream &in, size_t nfeatures) {
  auto data = std::vector<double>{};
  auto y = std::vector<double>{};
  // TODO: Use something faster than streams and getline?
  std::string line;
  int nsamples = 0;
  while (getline(in, line) && in.good()) {
    std::stringstream numberstr;
    size_t line_features = 0;
    auto c = line.begin();
    while (true) {
      if (*c == delimiter) {
        double number = std::stod(numberstr.str());
        numberstr.str("");
        if (line_features < nfeatures) {
          data.push_back(number);
        } else {
          y.push_back(number);
          break;
        }
        line_features++;
      }
      if (c == line.end()) {
        if (line_features < nfeatures) {
          // TODO: Add line number
          throw std::runtime_error("Insuficcient features");
        }
        double number = std::stod(numberstr.str());
        numberstr.str("");
        y.push_back(number);
        break;
      } else {
        numberstr << *c;
      }
      ++c;
    }
    nsamples++;
  }
  return {Data2D(nsamples, nfeatures, std::move(data)), std::move(y)};
}

// prefer to handle manually as cxxopt retains some references
// that I don't want to investigate.
struct args {
  std::string input;
  std::string output;
  size_t nfeatures;
  double lambda;
  double noise;
};

struct args parse(int argc, char *argv[]) {
  try {
    cxxopts::Options options(argv[0],
                             "Build an interpolation grid from a text file");
    options.positional_help("[optional args]").show_positional_help();

    options.add_options()("i,input", "Tab separated input file",
                          cxxopts::value<std::string>(),
                          "INPUT")("o,output", "Location of the output file",
                                   cxxopts::value<std::string>(), "OUTPUT")(
        "n,nfeatures", "Number of spatial features", cxxopts::value<size_t>(),
        "N")("l,lambda", "Correlation length", cxxopts::value<double>(),
             "L")("s,noise", "Noise level", cxxopts::value<double>(),
                  "S")("h,help", "Print help");

    options.parse_positional({"input", "output"});

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(0);
    }

    std::string input;
    std::string output;

    if (result.count("input")) {
      input = result["input"].as<std::string>();
    } else {
      std::cerr << "Expecting input location.\n";
      exit(1);
    }

    if (result.count("output")) {
      output = result["output"].as<std::string>();
    } else {
      std::cerr << "Expecting output location.\n";
      exit(1);
    }

    size_t nfeatures;
    if (result.count("nfeatures")) {
      nfeatures = result["nfeatures"].as<std::size_t>();
    } else {
      std::cerr << "Number of features must be specified\n";
      exit(1);
    }

    double lambda;
    if (result.count("lambda")) {
      lambda = result["lambda"].as<double>();
    } else {
      lambda = 0.05;
    }

    double noise;
    if (result.count("noise")) {
      noise = result["noise"].as<double>();
    } else {
      noise = 1e-10;
    }

    return {input, output, nfeatures, lambda, noise};

  } catch (const cxxopts::OptionException &e) {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}

int main(int argc, char **argv) {
  auto args = parse(argc, argv);
  std::ifstream in(args.input);
  if (in.fail()) {
    std::cerr << "Cannot open input file " << args.input;
    exit(1);
  }

  auto [data, y] = parse_csv(in, args.nfeatures);
  auto tree =
      ZKDTree::KDTree(std::move(data), std::move(y), args.lambda, args.noise);
  ZKDTree::save_tree(tree, args.output.c_str());

  return 0;
}
