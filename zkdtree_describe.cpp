#include "tree.hpp"

#include "cxxopts.hpp"

#include <iostream>
#include <string>
#include <variant>

struct args {
  std::string tree_input;
  std::string dump_path;
};

struct args parse(int argc, char **argv) {
  try {
    cxxopts::Options options(argv[0], "Print parameters of input tree");
    options.positional_help("[optional args]").show_positional_help();

    // clang-format off
    options.add_options()
		("i,input", "Tab separated input file", cxxopts::value<std::string>(), "INPUT")
		("d,dump", "Optional file path to dump the data to", cxxopts::value<std::string>(), "DUMP")
		("h,help", "Print help");
    // clang-format on
    options.parse_positional({"input"});

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help({"", "Group"}) << std::endl;
      exit(0);
    }

    auto input = std::string{};
    auto dump_path = std::string{};

    if (result.count("input")) {
      input = result["input"].as<std::string>();
    } else {
      std::cerr << "Expecting input location.\n";
      exit(1);
    }
    if (result.count("dump")) {
      dump_path = result["dump"].as<std::string>();
    }
    auto res = args{input, dump_path};
    return res;
  } catch (const cxxopts::OptionException &e) {
    std::cerr << "error parsing options " << e.what() << std::endl;
    exit(1);
  }
}

int main(int argc, char **argv) {
  auto args = parse(argc, argv);
  auto v = ZKDTree::load_tree(args.tree_input.c_str());
  if (std::holds_alternative<std::string>(v)) {
    std::cerr << std::get<std::string>(v);
    exit(1);
  }
  auto tree = std::get<ZKDTree::KDTree>(v);
  std::cout << "rbf_scale: " << tree.rbf_scale << "\n";
  std::cout << "noise_scale: " << tree.noise_scale << "\n";
  std::cout << "kernel_scale: " << tree.kernel_scale << "\n";
  std::cout << "nneighbours: " << tree.nneighbours << "\n";
  std::cout << "nsamples: " << tree.nsamples() << "\n";
  std::cout << "nfeatures: " << tree.nfeatures() << "\n";
  if (!args.dump_path.empty()) {
    std::ofstream os(args.dump_path);
	if(!os.good()){
		std::cerr << "Cannot write to output file: " << args.dump_path << "\n";
	}
    for (size_t i = 0; i < tree.nsamples(); i++) {
      for (size_t j = 0; j < tree.nfeatures(); j++) {
        os << tree.data.at(i, j) << "\t";
      }
      os << tree.responses.at(i) << "\n";
    }
  }
}
