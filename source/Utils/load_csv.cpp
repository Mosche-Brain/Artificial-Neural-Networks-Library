// //#include "csv_utils.hpp"

// #include <Eigen/Dense>
// #include <fstream>
// #include <sstream>
// #include <vector>


// MatrixXd csv_to_matrix(std::string& data, char sep)
// {
//     std::vector<std::vector<std::string>> dataMatrix;

//     int row = 0;
//     int col = 0;
//     for(int i = 0 ; i < data.size() ; i++)
//     {
//         switch(data[i]
//         {
//             case '\n': 
//             { 
//                 dataMatrix.push_back(std::vector<std::string>(col));
                
//                 row++;
//                 col = 0;
//                 break;
//             }

//             case sep:
//             {
//                 col++;
//                 break; 
//             }

//             case ' ' : { break ; }
        
//             default:
//             {
//                 std::string literal;
//                 for(int j = 0 ; data[i + j] != sep ; j++)
//                 {
//                     literal += data[i + j]   
//                 }

//                 dataMatrix.at(row).at(col).push_back(literal);
//             }
//         }
//     }


// }
