from backend.image_validator import validate_skin_image


image_path = r"C:\Users\dhara\Downloads\skin.jpg"


result = validate_skin_image(image_path)

print("\n========== VALIDATION RESULT ==========")

for key, value in result.items():
    print(f"{key}: {value}")