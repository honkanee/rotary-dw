#include <H5Cpp.h>
#include <iostream>
#include <Kokkos_Core.hpp>

#include "hdf5writer.h"
#include "simulation_parameters.h"
#include "state.h"

// ----------------------
// HDF5 writer
// ----------------------
HDF5Writer::HDF5Writer(SimulationState& state, SimulationParameters& p, std::string& h5_directory)
        : s(state), p(p), h5_directory(h5_directory)
    {

    }

void HDF5Writer::make_output_file()
    {
        try {
            // Code to create the HDF5 file and datasets

            if (p.file_partioning) {
                output_file = H5::H5File(h5_directory+"/part_"+std::to_string(file_count)+".hdf5", H5F_ACC_TRUNC);
            }
            else
            {
                output_file = H5::H5File(h5_directory+"out.hdf5", H5F_ACC_TRUNC);
            }

            // Add attributes to the group
            H5::DataSpace attribute_dataspace(H5S_SCALAR); // Scalar dataspace for a single value
            H5::Attribute attribute = output_file.createAttribute("n", H5::PredType::NATIVE_INT, attribute_dataspace);
            int N = p.N;
            attribute.write(H5::PredType::NATIVE_INT, &N);

            double Bext = p.Bext_SI();
            H5::Attribute attribute_Bext = output_file.createAttribute("Bext_mT", H5::PredType::NATIVE_DOUBLE, attribute_dataspace);
            attribute_Bext.write(H5::PredType::NATIVE_DOUBLE, &Bext);

            double sigma = p.sigma_SI();
            H5::Attribute attribute_sigma = output_file.createAttribute("sigma_mT", H5::PredType::NATIVE_DOUBLE, attribute_dataspace);
            attribute_sigma.write(H5::PredType::NATIVE_DOUBLE, &sigma);

            double delta_t = p.dt_SI();
            H5::Attribute attribute_Delta_t = output_file.createAttribute("Delta_t_ns", H5::PredType::NATIVE_DOUBLE, attribute_dataspace);
            attribute_Delta_t.write(H5::PredType::NATIVE_DOUBLE, &delta_t);

            double dmi_const = p.dmi_const_SI();
            H5::Attribute attribute_dmi_const = output_file.createAttribute("DMI_const_J_per_m2", H5::PredType::NATIVE_DOUBLE, attribute_dataspace);
            attribute_dmi_const.write(H5::PredType::NATIVE_DOUBLE, &dmi_const);

            // Create a dataspace for a single row
            hsize_t rowDims[2] = {1, p.resolution_along_x}; // One row at a time
            H5::DataSpace rowSpace(1, rowDims);
            hsize_t dims[2]       = {1, p.resolution_along_x}; // dataset dimensions at creation
            hsize_t maxdims[2]    = {H5S_UNLIMITED, p.resolution_along_x};
            H5::DataSpace *dataspace = new H5::DataSpace(2, dims, maxdims);
            H5::DataSpace *dataspace_phi = new H5::DataSpace(2, dims, maxdims);

            // Create a chunked dataset with an unlimited dimension for resizable rows
            H5::DSetCreatPropList createParams;
            createParams.setChunk(2, dims); // Each chunk contains one row

            h_dataset = output_file.createDataSet("h", H5::PredType::NATIVE_DOUBLE, *dataspace, createParams);

            H5::StrType attr_type(H5::PredType::C_S1, 256); // 256 is the maximum string length
            H5::Attribute attribute_h_unit = h_dataset.createAttribute("unit", attr_type, attribute_dataspace);
            attribute_h_unit.write(attr_type, "um");

            phi_dataset = output_file.createDataSet("phi", H5::PredType::NATIVE_DOUBLE, *dataspace_phi, createParams);

            // Create a one-dimensional datasets v and B_ext
            if (p.ramped_field) {
                hsize_t dims_Bext[1]  = {1};
                hsize_t maxdims_Bext[1]    = {H5S_UNLIMITED};
                H5::DataSpace rowSpace_Bext(1);
                H5::DataSpace *dataspace_Bext = new H5::DataSpace(1, dims_Bext, maxdims_Bext);

                H5::DSetCreatPropList createParams_Bext;
                createParams_Bext.setChunk(1, dims_Bext);
                Bext_dataset = output_file.createDataSet("Bext", H5::PredType::NATIVE_DOUBLE, *dataspace_Bext, createParams_Bext);
            }

            hsize_t dims_v_mean[1]  = {1};
            hsize_t maxdims_v_mean[1]    = {H5S_UNLIMITED};
            H5::DataSpace rowSpace_v_mean(1);
            H5::DataSpace *dataspace_v_mean = new H5::DataSpace(1, dims_v_mean, maxdims_v_mean);

            H5::DSetCreatPropList createParams_v_mean;
            createParams_v_mean.setChunk(1, dims_v_mean);
            v_mean_dataset = output_file.createDataSet("v_mean", H5::PredType::NATIVE_DOUBLE, *dataspace_v_mean, createParams_v_mean);

            rows = 0;


        } catch (const H5::FileIException& error) {
            std::cerr << "HDF5 File Exception: " << error.getDetailMsg() << std::endl;
        } catch (const H5::DataSetIException& error) {
            std::cerr << "HDF5 DataSet Exception: " << error.getDetailMsg() << std::endl;
        }
    }
    
void HDF5Writer::write(int step_count)
    {
        hsize_t size[2];
        size[0] = rows+1;
        size[1] = p.resolution_along_x;

        hsize_t rowDims[2] = {1,p.resolution_along_x};


        ViewComplexVectorType coarse_dw("Coarse grained DW", p.resolution_along_x);
        coarse_dw = spatial_average(s.dw, p.resolution_along_x);
        auto h_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coarse_dw);

        for (size_t i = 0; i < h_host.extent(0); ++i) {
            if (!std::isfinite(h_host(i).real()) || !std::isfinite(h_host(i).imag())) {
                std::cout << "BAD VALUE BEFORE HDF5 at " << i << "\n";
            }
        }

        std::vector<double> h_in_um(h_host.extent(0));
        std::vector<double> phi(h_host.extent(0));
        for (size_t i = 0; i < h_host.extent(0); ++i) {
            h_in_um[i] = h_host(i).real() * p.length_scale() * 1e6;
            phi[i] = -1 * h_host(i).imag();
        }

        h_dataset.extend(size);
        phi_dataset.extend(size);
        hsize_t ZERO = 0;
        hsize_t offset[2] = {rows, ZERO};

        H5::DataSpace *h_filespace = new H5::DataSpace(h_dataset.getSpace());
        h_filespace->selectHyperslab(H5S_SELECT_SET, rowDims, offset);
        H5::DataSpace *memspace = new H5::DataSpace(2, rowDims, NULL);
        h_dataset.write(h_in_um.data(), H5::PredType::NATIVE_DOUBLE, *memspace, *h_filespace);
        delete h_filespace;
        delete memspace;

        H5::DataSpace *phi_filespace = new H5::DataSpace(phi_dataset.getSpace());
        phi_filespace->selectHyperslab(H5S_SELECT_SET, rowDims, offset);
        H5::DataSpace *phi_memspace = new H5::DataSpace(2, rowDims, NULL);
        phi_dataset.write(phi.data(), H5::PredType::NATIVE_DOUBLE, *phi_memspace, *phi_filespace);
        delete phi_filespace;
        delete phi_memspace;

        if (p.ramped_field)
        {
            // TODO: Modify for the new implementation
            //hsize_t size[1] = {rows*p.calc_v_frec+p.calc_v_frec};
            //Bext_dataset.extend(size);
            //hsize_t rowDims_Bext[1] = {1};
            //H5::DataSpace *Bext_filespace = new H5::DataSpace(Bext_dataset.getSpace());
            //hsize_t offset[1] = {rows};
            //Bex//t_filespace->selectHyperslab(H5S_SELECT_SET, rowDims_Bext, offset);
            //H5::DataSpace *Bext_memspace = new H5::DataSpace(1, rowDims_Bext, NULL);
            //double Bext_mT = *B_ptr*(*m_dw).mu0*(*m_dw).Ms*1000;
            //Bext_dataset.write(&Bext_mT, H5::PredType::NATIVE_DOUBLE, *Bext_memspace, *Bext_filespace);
            //delete Bext_memspace;
            //delete Bext_filespace;
        }

        std::vector<double> v_mean_in_m_per_s(s.v_idx);
        auto host_v = Kokkos::create_mirror_view(s.v_mean_vector);
        Kokkos::deep_copy(host_v, s.v_mean_vector);
        for (size_t i = 0; i < s.v_idx; ++i) {
            v_mean_in_m_per_s[i] = host_v(i) * p.length_scale() / p.time_scale();
        }

        hsize_t rows_v_per_write = int(s.v_idx);
        hsize_t current_size;
        v_mean_dataset.getSpace().getSimpleExtentDims(&current_size);
        hsize_t new_size = current_size + rows_v_per_write;
        v_mean_dataset.extend(&new_size);
        hsize_t rowDims_v_mean[1] = {rows_v_per_write};
        H5::DataSpace *v_mean_filespace = new H5::DataSpace(v_mean_dataset.getSpace());
        hsize_t offset_v[1] = {current_size};
        v_mean_filespace->selectHyperslab(H5S_SELECT_SET, rowDims_v_mean, offset_v);
        H5::DataSpace *v_mean_memspace = new H5::DataSpace(1, rowDims_v_mean, NULL);
        v_mean_dataset.write(v_mean_in_m_per_s.data(), H5::PredType::NATIVE_DOUBLE, *v_mean_memspace, *v_mean_filespace);
        delete v_mean_memspace;
        delete v_mean_filespace;

        ++rows;

        std::chrono::duration<double> run_time = std::chrono::system_clock::now()-run_clock_start;


        if (p.file_partioning && (step_count % (p.print_frec*p.num_rows_per_file)) == 0 && step_count!=0)
        {
            std::cout << "New file!" << std::endl;
            close();
            make_output_file();
            ++file_count;
        }
    }
    



void HDF5Writer::close()
    {
        output_file.close();
    }