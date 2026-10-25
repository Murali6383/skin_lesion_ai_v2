from ai.image_validator.validator import validate_skin_image


def validate_image(image_path: str):
    """
    Validate uploaded image using the non-skin nearest-neighbour validator.
    """

    print("\n------------------------------------------")
    print("RUNNING SKIN IMAGE VALIDATOR")
    print("------------------------------------------")

    result = validate_skin_image(image_path)

    print("\nValidator output:")
    print(result)

    return result


__all__ = [
    "validate_image",
    "validate_skin_image",
]