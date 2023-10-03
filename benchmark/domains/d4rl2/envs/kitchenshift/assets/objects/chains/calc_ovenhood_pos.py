ovr = [0.115, -0.2921, 0.9834]
hr = [0.1, 0.188, 2.33]

knobs = {
    1: [-0.148, 0.22, 1.243],  # botright
    2: [-0.271, 0.22, 1.243],  # botleft
    3: [-0.148, 0.22, 1.357],  # topright
    4: [-0.271, 0.22, 1.357],  # topleft
}

for n in range(1, 4 + 1):
    p = [ovr[i] - hr[i] + knobs[n][i] for i in range(3)]
    print('knob', n, p)

    #
    # # order matters, flip b/c original had knob body before burner body
    # 'bottomknob': np.array([14, 10]),  # bottom burner, left
    # 'topknob': np.array([16, 12]),  # top burner, right
    # 'bottomknobr': np.array([13, 9]),  # bottom burner, right
    # 'topknobr': np.array([15, 11]),  # top burner, right

ol = [0, 0.2, 2.25]
p = [hr[i] - ol[i] for i in range(3)]
print('ol', p)
