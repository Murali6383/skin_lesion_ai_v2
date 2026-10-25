from ai.normal_skin_detector.detector import detect_normal_skin


IMAGE_PATH = "ai/test_image.jpg"


result = detect_normal_skin(
    IMAGE_PATH
)


print("\n")
print("=" * 60)
print("NORMAL SKIN DETECTOR TEST")
print("=" * 60)

for key, value in result.items():

    print(
        f"{key}: {value}"
    )

print("=" * 60)