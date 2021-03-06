#define GLM_ENABLE_EXPERIMENTAL 1

#include <iostream>

#include <GL/glew.h>

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/Main.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/ext.hpp>
#include <glm/gtx/string_cast.hpp>

//#include <gl/GL.h>
//#include <gl/GLU.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <opengl/Camera.h>

using namespace std;

struct VertexData {
	glm::vec3 mPosition;
	glm::vec3 mColor;
};

GLuint CreateVertexBuffer(const std::vector<VertexData>& aData);
void RenderVertexBuffer(GLuint iVertexBuffer, int iVertexDataSize, GLenum eRenderMode, int iOffset, int iCount);
vector<VertexData> CreateGrid(int width, int height, int depth);

int main()
{
	unsigned int iWidth = 800;
	unsigned int iHeight = 600;
	sf::ContextSettings settings;
	settings.antialiasingLevel = 4;
	sf::RenderWindow window(sf::VideoMode(iWidth, iHeight), "My window", sf::Style::Default, settings);
	window.setVerticalSyncEnabled(true);
	window.setActive(true);

	glewExperimental = true; // Needed in core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		system("pause");
	}

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);


	sf::Shader oShader;
	if (!oShader.loadFromFile("VertexShader.glsl", "FragmentShader.glsl")) {
		cout << "Error loading shader" << endl;
	}


	vector<VertexData> aData = {
		{ { 0.0f, 0.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } }, // bottom 1
		{ { 1.0f, 0.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { 1.0f, 0.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },

		{ { 0.0f, 0.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } }, // bottom 2
		{ { 1.0f, 0.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { 0.0f, 0.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },

		{ { 0.0f, 1.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } }, // top 1
		{ { 1.0f, 1.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { 1.0f, 1.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },

		{ { 0.0f, 1.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } }, // top 2
		{ { 1.0f, 1.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { 0.0f, 1.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },

		{ { 0.0f, 0.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } }, // left 1
		{ { 0.0f, 0.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { 0.0f, 1.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },

		{ { 0.0f, 0.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } }, // left 2
		{ { 0.0f, 1.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { 0.0f, 1.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } },

		{ { 1.0f, 0.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } }, // right 1
		{ { 1.0f, 0.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { 1.0f, 1.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },

		{ { 1.0f, 0.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } }, // right 2
		{ { 1.0f, 1.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { 1.0f, 1.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } },

		{ { 0.0f, 0.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } }, // front 1
		{ { 1.0f, 0.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { 1.0f, 1.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } },

		{ { 0.0f, 0.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } }, // front 2
		{ { 1.0f, 1.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { 0.0f, 1.0f, 0.0f },{ 1.0f, 0.0f, 0.0f } },

		{ { 0.0f, 0.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } }, // back 1
		{ { 1.0f, 0.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { 1.0f, 1.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },

		{ { 0.0f, 0.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } }, // back 2
		{ { 1.0f, 1.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } },
		{ { 0.0f, 1.0f, 1.0f },{ 1.0f, 0.0f, 0.0f } }
	};

	GLuint vertexbuffer = CreateVertexBuffer(aData);

	vector<VertexData> aDataLines = CreateGrid(10, 10, 10);

	GLuint vertexbuffer2 = CreateVertexBuffer(aDataLines);

	Camera oCam;

	sf::Shader::bind(&oShader);

	sf::Vector2i oMouseOld = sf::Mouse::getPosition(window);
	while (window.isOpen())
	{
		// check all the window's events that were triggered since the last iteration of the loop
		sf::Event event;
		while (window.pollEvent(event))
		{
			// "close requested" event: we close the window
			if (event.type == sf::Event::Closed)
				window.close();
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {
			oCam.Move(1, 0);
		}
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
			oCam.Move(-1, 0);
		}
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
			oCam.Move(0, 1);
		}
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
			oCam.Move(0, -1);
		}

		sf::Vector2i oMouseNew = sf::Mouse::getPosition(window);
		if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {

			float dx = (float)(oMouseNew.x - oMouseOld.x);
			float dy = (float)(oMouseNew.y - oMouseOld.y);
			oCam.Rotate(-dy/500.0f, -dx/500.0f);
		}

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glm::mat4 Projection = glm::perspective(glm::radians(45.0f), (float)iWidth / (float)iHeight, 0.1f, 100.0f);

		/*glm::mat4 View = glm::lookAt(
			glm::vec3(4, 3, 3), // Camera is at (4,3,3), in World Space
			glm::vec3(0, 0, 0), // and looks at the origin
			glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
		);*/
		//glm::mat4 View = oCam.GetViewMatrix();
		/*glm::vec3 front(1.0, 0.0, 1.0);
		front = glm::normalize(front);
		front = glm::rotateY(front, RotY);
		glm::vec3 position(-3.0f, 0.0f, -3.0f);
		glm::vec3 lookat = position + front;
		glm::vec3 up(0.0f, 1.0f, 0.0f);
		glm::mat4 View = glm::lookAt(position, lookat, up);*/
		glm::mat4 View = oCam.getViewMatrix();


		glm::mat4 Model = glm::mat4(1.0f);
		glm::mat4 Model2 = glm::translate(Model, glm::vec3(2.0, 0.0, 0.0));

		glm::mat4 mvp = Projection * View * Model;
		glm::mat4 mvp2 = Projection * View * Model2;

		//GLuint MatrixID = glGetUniformLocation(oVertexShader.getNativeHandle(), "MVP");

		sf::Glsl::Mat4 omvp(glm::value_ptr(mvp));
		sf::Glsl::Mat4 omvp2(glm::value_ptr(mvp2));

		oShader.setUniform("MVP", sf::Glsl::Mat4(glm::value_ptr(mvp)));

		RenderVertexBuffer(vertexbuffer2, sizeof(VertexData), GL_LINES, 0, aDataLines.size());
		RenderVertexBuffer(vertexbuffer, sizeof(VertexData), GL_TRIANGLES, 0, aData.size());

		//oShader.setUniform("MVP", omvp2);
		//sf::Shader::bind(&oShader);
		mvp = Projection * View * Model2;
		oShader.setUniform("MVP", sf::Glsl::Mat4(glm::value_ptr(mvp)));
		//glUniformMatrix4fv(iMVPLoc, 1, GL_FALSE, glm::value_ptr(mvp));
		RenderVertexBuffer(vertexbuffer, sizeof(VertexData), GL_TRIANGLES, 0, aData.size());

		window.display();

		oMouseOld = oMouseNew;
	}

    return 0;
}

GLuint CreateVertexBuffer(const std::vector<VertexData>& aData) {
	GLuint iVertexBufferResult;
	// Generate 1 buffer, put the resulting identifier in vertexbuffer
	glGenBuffers(1, &iVertexBufferResult);
	// The following commands will talk about our 'vertexbuffer' buffer
	glBindBuffer(GL_ARRAY_BUFFER, iVertexBufferResult);
	// Give our vertices to OpenGL.
	glBufferData(GL_ARRAY_BUFFER, sizeof(VertexData)*aData.size(), aData.data(), GL_STATIC_DRAW);

	return iVertexBufferResult;
}

void RenderVertexBuffer(GLuint iVertexBuffer, int iVertexDataSize, GLenum eRenderMode, int iOffset, int iCount) {
	glBindBuffer(GL_ARRAY_BUFFER, iVertexBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, iVertexDataSize, (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, iVertexDataSize, (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(iVertexBuffer);

	glDrawArrays(eRenderMode, iOffset, iCount);

	glDisableVertexAttribArray(0);
}

vector<VertexData> CreateGrid(int width, int height, int depth) {
	vector<VertexData> aResult;

	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < height; ++j) {
			aResult.push_back({ {(float)(width / 2), (float)(j-height/2), (float)(i - depth / 2)}, {0.0, 0.0, 1.0} });
			aResult.push_back({ { (float)(-width / 2), (float)(j-height/2), (float)(i - depth / 2) },{ 0.0, 0.0, 1.0 } });
		}
	}

	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < height; ++j) {
			aResult.push_back({ { (float)(i-width / 2), (float)(j - height / 2), (float)(depth / 2) },{ 0.0, 0.0, 1.0 } });
			aResult.push_back({ { (float)(i-width / 2), (float)(j - height / 2), (float)(-depth / 2) },{ 0.0, 0.0, 1.0 } });
		}
	}

	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < height; ++j) {
			aResult.push_back({ { (float)(i - width / 2), (float)(height / 2), (float)(j-depth / 2) },{ 0.0, 0.0, 1.0 } });
			aResult.push_back({ { (float)(i - width / 2), (float)(-height / 2), (float)(j-depth / 2) },{ 0.0, 0.0, 1.0 } });
		}
	}

	return aResult;
}
