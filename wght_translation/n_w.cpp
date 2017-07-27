#include <iostream>
#include "Numpy.hpp"

int main()
{
	std::vector<int> s;
	std::vector<float> data;
	aoba::LoadArrayFromNumpy("array.npy", s, data);
	std::cout << s[0] << " " << s[1] <<" " << s[2] <<" " << s[3] << std::endl; 
	std::cout << data.size() << std::endl; 
	for (float n : data) std::cout << n << ' ';
	
}