// #pragma once 

// #include <Tensor.h>

// #include <sstream> 

// namespace coreten {
//     std::ostream& operator<<(std::ostream & out, Backend b) { return out << toString(b); }
// } //namespace coreten 


// namespace ad {
//     static void __printIndent(std::ostream &stream, int64_t indent) {
//         for(int64_t i = 0; i<indent; i++) {
//             stream << " ";
//         }
//     }

//     std::ostream& print(std::ostream& stream, const Tensor& tens, int64_t linesize) {
//     FormatGuard guard(stream);

//     if(!tens.defined()) { stream << "[ Tensor (undefined) ]"; } 

//     else {
//         Tensor tensor;

//         if(tensor.ndimen() == 0) {
//         // stream << defaultfloat << tensor.data_ptr<double>()[0] << std::endl;
//         stream << "[ " << tens.toString() << "{}";
//         } else if(tensor.ndimen() == 1) {
//         if (tensor.numel() > 0) {
//             double scale;
//             int64_t sz;
//             std::tie(scale, sz) =  __printFormat(stream, tensor);
//             if(scale != 1) {
//                 printScale(stream, scale);
//             }
//             double* tensp = tensor.data_ptr<double>();
//             for(int64_t i = 0; i < tensor.size(0); i++) {
//                 stream << std::setw(sz) << tensp[i]/scale << std::endl;
//             }
//         }
//         stream << "[ " << tens.toString() << "{" << tensor.size(0) << "}";
//         } else if(tensor.ndimen() == 2) {
//         if (tensor.numel() > 0) {
//             __printMatrix(stream, tensor, linesize, 0);
//         }
//         stream << "[ " << tens.toString() << "{" << tensor.size(0) << "," <<  tensor.size(1) << "}";
//         } else {
//         if (tensor.numel() > 0) {
//             __printTensor(stream, tensor, linesize);
//         }
//         stream << "[ " << tens.toString() << "{" << tensor.size(0);
//         for(int64_t i = 1; i < tensor.ndimension(); i++) {
//             stream << "," << tensor.size(i);
//         }
//         stream << "}";
//         }
//         if (tens.is_quantized()) {
//         stream << ", qscheme: " << toString(tens.qscheme());
//         if (tens.qscheme() == coreten::kPerTensorAffine) {
//             stream << ", scale: " << tens.q_scale();
//             stream << ", zero_point: " << tens.q_zero_point();
//         } else if (tens.qscheme() == coreten::kPerChannelAffine ||
//             tens.qscheme() == coreten::kPerChannelAffineFloatQParams) {
//             stream << ", scales: ";
//             Tensor scales = tens.q_per_channel_scales();
//             print(stream, scales, linesize);
//             stream << ", zero_points: ";
//             Tensor zero_points = tens.q_per_channel_zero_points();
//             print(stream, zero_points, linesize);
//             stream << ", axis: " << tens.q_per_channel_axis();
//         }
//         }

//         auto& fw_grad = tensor.fw_grad(/* level */ 0);
//         if (fw_grad.defined()) {
//         stream << ", tangent:" << std::endl << fw_grad;
//         }
//         stream << " ]";
//     }
//     return stream;
//     }



// } //namespace ad 
