#pragma once

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Camera
{
public:
	Camera(glm::vec3 Position = glm::vec3(0.0f, 0.5f, -1.0f), glm::vec3 Up = glm::vec3(0.0, 1.0, 0.0), float RotY = 0.0f, float RotX = 0.0f);
	virtual ~Camera();

	void Move(int dx, int dy);
	void Rotate(float dx, float dy);

	glm::mat4 getViewMatrix() const;
private:
	glm::mat4 mViewMatrix;

	glm::vec3 mPosition;
	glm::vec3 mFront;
	glm::vec3 mRight;
	glm::vec3 mUp;

	float mRotationY;
	float mRotationX;
};

