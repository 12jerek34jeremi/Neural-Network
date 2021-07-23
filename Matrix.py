import math


class Matrix(list):
    def __init__(self, n_rows, n_columns, creator=None):
        """Argument 'creator' is supposed to be a function which returns a number (float or int)
            This constructor use this function to initialize each element of matrix.
            For example if you pass such a function: 'lambda : return random.randrange(1,6)'
             each element of matrix is number from set {1, 2, 3, 4, 5}
             If creator isn't passed each element of a matrix is zero (0.0).
             Rest is self explanatory"""
        super().__init__()
        self.n_rows = n_rows
        self.n_columns = n_columns
        for row in range(n_rows):
            self.append([])
            for column in range(n_columns):
                if creator:
                    self[row].append(creator())
                else:
                    self[row].append(0.0)

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise Exception("Adding matrix to not Matrix.")

        if self.n_columns != other.n_columns and self.n_rows != other.n_rows:
            raise Exception("Impossible to add this two matrices.")

        result = Matrix(self.n_rows, self.n_columns)
        for i in range(self.n_rows):
            for j in range(self.n_columns):
                result[i][j] = self[i][j] + other[i][j]

        return result

    def __iadd__(self, other):
        result = self.__add__(other)
        return result

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.n_columns != other.n_rows:
                raise Exception("Impossible to multiply this two matrices!")
            result = Matrix(self.n_rows, other.n_columns)
            for rowM1 in range(self.n_rows):
                for columnM2 in range(other.n_columns):
                    sum = 0
                    for i in range(self.n_columns):
                        sum += self[rowM1][i] * other[i][columnM2]
                    result[rowM1][columnM2] = sum
            result.n_rows = len(result)
            result.n_columns = len(result[0])
            return result
        elif type(other) == int or type(other) == float:
            result = Matrix(self.n_rows, self.n_columns)
            for i in range(self.n_rows):
                for j in range(self.n_columns):
                    result[i][j] = self[i][j] * other
            return result
        else:
            raise Exception("Matrix can be multiplied only by another Matrix or number\n"
                            "Type of other is: ", type(other))

    def __imul__(self, other):
        print("I'm working!")
        result = self.__mul__(other)
        return result

    @staticmethod
    def sigma(matrix):
        """This method run an sigma function (mathematics function, i won't find it in this code) on every element of
        given matrix and return new matrix with changed elements. Each element of new matrix is in interval (0, 1).
        This method isn't changing given matrix."""
        result = Matrix(matrix.n_rows, 1)
        for row in range(matrix.n_rows):
            result[row][0] = 1 / (1 + math.exp(-matrix[row][0]))
        return result

    @staticmethod
    def sigma_derivative(a):
        """This method is counting sigma derivatives for a specific layer and returning matrix representing them.
            Shape of returned matrix is (n, 1), where n i number of neurons in given layer."""
        matrix_rigth = Matrix(a.n_rows, 1)
        for i in range(a.n_rows):
            matrix_rigth[i][0] = 1 - a[i][0]
        return Matrix.hadamar_product(a, matrix_rigth)

    @staticmethod
    def hadamar_product(matrix1, matrix2):
        """This method is counting a hadamar product of two given matrices. Hadamar product is operation which crates
            new matrix of the same shape as two given matrix in which each component is eqaul to multuplication
            of two components from input matrices of the same indexes."""
        if matrix1.n_rows != matrix2.n_rows or matrix1.n_columns != matrix2.n_columns:
            raise Exception("Imposible to make hadamar product of these two matrices!")
        result = Matrix(matrix1.n_rows, matrix1.n_columns)
        for i in range(matrix1.n_rows):
            for j in range(matrix1.n_columns):
                result[i][j] = matrix1[i][j] * matrix2[i][j]
        result.n_rows = len(result)
        result.n_columns = len(result[0])
        return result

    def __sub__(self, other):
        if isinstance(other, Matrix):
            if self.n_rows != other.n_rows or self.n_columns != other.n_columns:
                raise Exception("Imposible to substruct these two matrices!")
            result = Matrix(self.n_rows, self.n_columns, lambda: 0)
            for i in range(self.n_rows):
                for j in range(self.n_columns):
                    result[i][j] = self[i][j] - other[i][j]
            return result
        else:
            raise Exception("Error, substrycting non matrix from matrix\n"
                            "Type of other is: ", type(other))

    def __isub__(self, other):
        return self.__sub__(other)

    def __str__(self):
        result = "--- --- --- ---\n["
        for row in self:
            result += "["
            for number in row:
                result += str(number) + ", "
            result = result[:-2] + "]\n"
        result = result[:-1] + "]\n--- --- --- ---"
        return result

    def transpose(self):
        """This function doesn't change an object! It is creating new transposed matrix and returning it."""
        result = Matrix(self.n_columns, self.n_rows, lambda: 0)
        for i in range(result.n_rows):
            for j in range(result.n_columns):
                result[i][j] = self[j][i]
        return result

    def copy(self, matrix):
        """This method copy given matrix to object it is ran on. (sth like private method)"""
        self.clear()
        for i in range(matrix.n_rows):
            self.append([])
            for j in range(matrix.n_columns):
                self[i].append(matrix[i][j])
        self.n_rows = matrix.n_rows
        self.n_columns = matrix.n_columns
