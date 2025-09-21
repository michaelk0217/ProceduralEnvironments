slangc shader.slang -target spirv -entry vertexMain -o vert.spirv
slangc shader.slang -target spirv -entry fragmentMain -o frag.spirv
slangc heightmap.slang -target spirv -entry main -o heightmap.spirv
pause