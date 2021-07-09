import math

class Matrice(list):
    def __init__(self, n_rows, n_columns, creator):
        "If first parameter (not counting self) is 0 concstructor does not do anythink"
        if n_rows != 0:
            self.n_rows = n_rows
            self.n_columns = n_columns
            for row in range(n_rows):
                self.append([])
                for column in range(n_columns):
                    self[row].append(creator())

    def __mul__(self, other):
        if isinstance(other, Matrice):
            if self.n_columns != other.n_rows:
                raise Exception("Impossible to multiply this two matrices!")
                return None
            result = Matrice(0, 0, None)
            for rowM1 in range(self.n_rows):
                result.append([])
                for columnM2 in range(other.n_columns):
                    sum = 0
                    for i in range(self.n_columns):
                        sum += self[rowM1][i] * other[i][columnM2]
                    result[rowM1].append(sum)
            result.n_rows = len(result)
            result.n_columns = len(result[0])
            return result
        else:
            raise Exception("Error, multyplying matrice by non matrice")


    @staticmethod
    def sigma(matrice):
        result = Matrice(matrice.n_rows, 1, lambda : 0)
        for row in range(matrice.n_rows):
            result[row][0] = 1 / (1 + math.exp(-matrice[row][0]))
        return result

    @staticmethod
    def hadamar_product(matrice1, matrice2):
        if matrice1.n_rows != matrice2.n_rows or matrice1.n_columns != matrice2.n_columns:
            raise Exception("Imposible to make hadamar product of these two matrices!")
            return None
        result = Matrice(0, 0, None)
        for i in range(matrice1.n_rows):
            result.append([])
            for j in range(matrice1.n_columns):
                result.append(matrice1[i][j] * matrice2[i][j])
        result.n_rows = len(result)
        result.n_columns = len(result[0])
        return result

    def __sub__(self, other):
        if(isinstance(other, Matrice)):
            if self.n_rows != other.n_rows or self.n_columns != other.n_columns:
                raise Exception("Imposible to substruct these two matrices!")
                return None
            result = Matrice(self.n_rows, self.n_columns, lambda : 0)
            for i in range(self.n_rows):
                for j in range(self.n_columns):
                    result[i][j] = self[i][j] - other[i][j]
            return result
        else:
            raise Exception("Error, substrycting non matrice from matrice")

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
        result = Matrice( self.n_columns, self.n_rows, lambda : 0)
        for i in range(self.n_rows):
            for j in range(self.n_columns):
                result[i][j] = self[j][i]
        self.copy(result)

    def __copy(self, matrice):
        for i in range(self.n_rows):
            for j in range(self.n_columns):
                self[i][j] = matrice[i][j]

