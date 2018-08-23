#include "Camera.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/rotate_vector.hpp>

Camera::Camera(glm::vec3 Position, glm::vec3 Up, float RotY, float RotX) :
	mPosition(Position),
	mFront(0.0, 0.0, 1.0),
	mRight(1.0f, 0.0f, 0.0f),
	mUp(Up),
	mRotationY(RotY),
	mRotationX(RotX) {
	
	mViewMatrix = glm::lookAt(mPosition, mPosition + mFront, mUp);

}


Camera::~Camera()
{
}

void Camera::Move(int dx, int dy) {
	if (dx > 0) {
		mPosition += 0.1f*mFront;
	}
	else if (dx < 0) {
		mPosition -= 0.1f*mFront;
	}

	if (dy > 0) {
		mPosition += 0.1f*mRight;
	}
	else if (dy < 0) {
		mPosition -= 0.1f*mRight;
	}

	mViewMatrix = glm::lookAt(mPosition, mPosition + mFront, mUp);
}

void Camera::Rotate(float dx, float dy) {
	if (dx == 0.0f && dy==0.0f)		return;

	mRotationX += dx;
	mRotationY += dy;
	mFront = glm::vec3(0.0, 0.0, 1.0);
	mFront = glm::rotateY(mFront, mRotationY);
	mFront = glm::rotate(mFront, mRotationX, glm::cross(mFront, glm::vec3(0.0, 1.0, 0.0)));
	mFront = glm::normalize(mFront);

	mRight = glm::cross(mFront, glm::vec3(0.0, 1.0, 0.0));
	mRight = -glm::normalize(mRight);
	mUp = glm::cross(mFront, mRight);

	mViewMatrix = glm::lookAt(mPosition, mPosition + mFront, mUp);
}

glm::mat4 Camera::getViewMatrix() const {
	return mViewMatrix;
}