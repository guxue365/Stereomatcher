#version 330 core
layout (location = 0) in vec3 aPos;   // the position variable has attribute position 0
layout (location = 1) in vec3 aColor; // the color variable has attribute position 1
  
out vec3 ourColor; // output a color to the fragment shader

// Values that stay constant for the whole mesh.
uniform mat4 MVP;
  
void main(){
  // Output position of the vertex, in clip space : MVP * position
  gl_Position =  MVP * vec4(aPos, 1);
  ourColor = aColor; // set ourColor to the input color we got from the vertex data
}