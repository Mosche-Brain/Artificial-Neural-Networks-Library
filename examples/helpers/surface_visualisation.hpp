//
// Created by jaro on 7/5/26.
//
#pragma once
#include <vector>
#include <easy3d/util/initializer.h>
#include <easy3d/viewer/viewer.h>
#include <easy3d/core/surface_mesh.h>
#include <easy3d/core/types.h>

using SurfaceFunc = std::function<cum::cumeric_t(cum::cumeric_t x, cum::cumeric_t y)>;

easy3d::SurfaceMesh* make_function_surface(
    const SurfaceFunc& func,
    cum::cumeric_t xmin, cum::cumeric_t xmax,
    cum::cumeric_t ymin, cum::cumeric_t ymax,
    int nx, int ny
) {
    auto* mesh = new easy3d::SurfaceMesh;

    std::vector<std::vector<easy3d::SurfaceMesh::Vertex>> vertices(
        nx + 1,
        std::vector<easy3d::SurfaceMesh::Vertex>(ny + 1)
    );

    for (int i = 0; i <= nx; ++i) {
        const cum::cumeric_t x = xmin + (xmax - xmin) * static_cast<cum::cumeric_t>(i) / static_cast<cum::cumeric_t>(nx);

        for (int j = 0; j <= ny; ++j) {
            const cum::cumeric_t y = ymin + (ymax - ymin) * static_cast<cum::cumeric_t>(j) / static_cast<cum::cumeric_t>(ny);

            const cum::cumeric_t z_value = func(
                static_cast<cum::cumeric_t>(x),
                static_cast<cum::cumeric_t>(y)
            );

            const cum::cumeric_t z = static_cast<float>(z_value);

            vertices[i][j] = mesh->add_vertex(easy3d::vec3(x, y, z));
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const auto v00 = vertices[i][j];
            const auto v10 = vertices[i + 1][j];
            const auto v01 = vertices[i][j + 1];
            const auto v11 = vertices[i + 1][j + 1];

            mesh->add_triangle(v00, v10, v11);
            mesh->add_triangle(v00, v11, v01);
        }
    }

    return mesh;
}
