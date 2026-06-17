def calculate_ttc(distance, speed):
    if speed <= 0:
        return float('inf')

    return distance / speed


distance = 20
speed = 10

ttc = calculate_ttc(distance, speed)

print("TTC =", ttc, "seconds")