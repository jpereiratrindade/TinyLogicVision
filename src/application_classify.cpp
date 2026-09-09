#include "tinyvision/application.hpp"
#include "tinyvision/image.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: tinyvision_classify MODEL.tlv IMAGE.png|IMAGE.jpg\n";
        return 2;
    }

    try {
        auto model = tinyvision::load_application_model(argv[1]);
        const auto input = tinyvision::resize_rgb_bilinear(
            tinyvision::load_rgb_image(std::filesystem::path(argv[2])),
            model.image_width,
            model.image_height);
        const auto probabilities = model.network.predict(input);
        const auto predicted = static_cast<std::size_t>(std::distance(
            probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end())));

        std::cout << "predicted=" << model.class_names[predicted] << '\n'
                  << "score=" << std::fixed << std::setprecision(6)
                  << probabilities[predicted] << '\n';
        for (std::size_t index = 0; index < probabilities.size(); ++index) {
            std::cout << "probability[" << model.class_names[index] << "]="
                      << probabilities[index] << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "tinyvision_classify: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
