slangc shader.slang -target spirv -entry vertexMain -o vert.spirv
slangc shader.slang -target spirv -entry fragmentMain -o frag.spirv
slangc heightmap.slang -target spirv -entry main -o heightmap.spirv
slangc height_normals.slang -target spirv -entry main -o heightmap_normals.spirv
slangc GenerateTerrainMesh.slang -target spirv -entry main -o GenerateTerrainMesh.spirv
pause