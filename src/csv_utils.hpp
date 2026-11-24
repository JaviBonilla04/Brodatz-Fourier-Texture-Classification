#ifndef CSV_UTILS_HPP
#define CSV_UTILS_HPP

#include <charconv>
#include <string_view>
#include <vector>

// Divide una línea en columnas usando el delimitador (coma por defecto).
// Las vistas apuntan a la línea original: la línea debe seguir viva mientras se usen.
inline std::vector<std::string_view> divide_line(std::string_view line, char delimitator = ','){
	std::vector<std::string_view> cols;
	size_t first = 0, end = line.find(delimitator);
	while(end != std::string_view::npos){
		cols.emplace_back(line.substr(first, end - first));
		first = end + 1;
		end = line.find(delimitator, first);
	}
	cols.emplace_back(line.substr(first));
	return cols;
}

// Convierte texto a número (int, float, double...).
template <typename T>
T toNum(std::string_view sv){
	T value{};
	std::from_chars(sv.data(), sv.data() + sv.size(), value);
	return value;
}

#endif