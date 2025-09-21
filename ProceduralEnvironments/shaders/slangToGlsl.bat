slangc heightmap.slang -target glsl -entry main -o heightmap.comp
glslang -V heightmap.comp -o heightmap.spirv -g
pause