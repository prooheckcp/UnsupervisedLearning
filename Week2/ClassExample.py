import tensorflow as tf

w = tf.Variable(3.0)
x = 1.0
y = 1.0 # target value
alpha = 0.01

iterations = 30

for iter in range(iterations):
    with tf.GradientTape() as tape:
        fwb = w * x
        costJ = (fwb - y)**2

    [dJdw] = tape.gradient(costJ, [w])

    w.assign_add(-alpha * dJdw)


class Solution:
    dict = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }

    def romanToInt(self, s: str) -> int:
        finalInteger = 0
        for i in range(len(s)):
            finalInteger = finalInteger + self.dict[s[i]]

        return finalInteger

solution = Solution()
solution.romanToInt('IV')