#ifndef IntersectionInfo_h_
#define IntersectionInfo_h_

struct BVHObject;

struct IntersectionInfo {
	float t; // Intersection distance along the ray
	const BVHObject* object; // BVHObject that was hit
	Vector3 hit; // Location of the intersection
};

#endif
