# Neural Network

Projekt Neural Network to implementacja sztucznej sieci neuronowej.

W tym repozytorium ważne są tak naprawdę dwa pliki:
1) Network.py:<br/>
Znajduje się w nim klasa Network będąca implementacją sieci neuronowej, pozwalającej na utworzenie sieci neuronwej z warstwami gęstymi (Fully Connected Layer). Ilość warst w sieci oraz ilość neuronów w każdej warstwie określa się przez argument konstruktora. Klasa implementuje też propagację wsteczną, umożliwia trenowanie sieci i zastosowanie takich regularyzacji jak DropDown, L1/L2 regularisation. Klasa umożliwia też zapisywane sieci w pliku i wczytawanie sieci z pliku.
2) MNIST_loader.py <br/>
W pliku tym znajdują się funkcje pozwalające na załadowanie danych do trenowania i testowania z plików, znajdujących się w katalogu "MNIST", do obiektów ndarray (do tablic numpy). Dane te to MNIST (obrazy ręcznie pisanych cyfr oraz odpowiadające im cyfry).

Jeżeli zamiast czytania pliku REDME.md wolisz czytać notebooka, to możesz przeczytać [ten notebook](notebook.ipynb). Treść tego notebooka jest prawie taka sama jak treść tego pliku.


Instrukcja używania:<br/>
Najpierw należy zaimportować wszystkie potrzebne rzeczy.
~~~ python
from Network import Network
import MNIST_loader as MN
~~~
Teraz należy załadować dane. Można do tego użyć poniższych dwóch funkcji z pliku MNIST_lodaer.
~~~ python
train_data = MN.load_training_tuples()
test_data = MN.load_test_tuples()
~~~
Funkcja load_training_tuples zwraza listę tupli (list[tuple[np.ndarray, int]). Pierwszy element każdego tupla to tablica numpy reprezentująca obraz przedstawiający daną cyfrę, drugi element tupli to int będący tą cyfrą. Dokładnie takiej listy będzie wymagała funkcja trenująca sieć.
Funkcja load_test_tuples robi to samo co funkcja load_training_tuples, ale zwraca dane do testowania. (Zbiór MNIST definiuje 60 tyś przykładów do trenowania i 10 tyś przykładów do testowania.)

Do wyświetlenia obrazu można użyć funkcji ‘show_image’.<br/>
![screen1](img/screen1.png)<br/>
---
Utwórzmy prostą sieć:
~~~ python
my_net = Network(shape=(784, 512, 1024, 2048, 10), name="net1")
~~~
Konstruktor wymaga podania kształtu sieci - informacji o tym, ile ma być warstw oraz ile ma być w każdej warstwie neuronów. Opcjonalnie można podać także nazwę sieci.

Do testowania sieci służy metoda `test_net`. Zwraca ona liczbę wszystkich poprawnie zakwalifikowanych przykładów podzieloną przez liczbę wszystkich przykładów.<br/>
![screen2](img/screen2.png)<br/>

Teraz można zacząć trenować sieć. Służy do tego funkcja 'train'.
~~~ python
my_net.train(epoch=10, mini_batch_size=30, eta=0.3, cost_function=1,train_data=train_data,
             test_data=test_data, dropout=False, L2_regularization_prm=0.1, L1_regularization_prm=None)
~~~
- Argument epoch oznacza liczbę epok do wykonania.
- Argument mini_batch_size określa wielkość mini batcha
- Argument eta oznacza learning_rate i określa jej wartość.
- Argument cost_function może przyjmować dwie wartośc. Gdy przujmuje on wartość
1 - do trenowania zostania użyta kwadratowa funkcja kosztu (MSE cost function)
2 -do trenowania zostanie użyta etropiczna funkcja kosztu (entropy cost function)
- Argument test_data jest opcjonalny. Podany oznacza dane, na których sieć będzie testowana po każdej epoce. W przypadku braku tego argumentu (lub jego wartości równej None), sieć nie będzie testowana w trakcie trenowania.
- Argument drop_out oznacza czy podczas trenowania ma zostać użyty dropdown. W przypadku wartości True dropout zostanie zastosowany, w przypadku wartości False dropout nie zostanie zastosowany. Domyślna wartość: False.<br/>
- Argument L1_regularization_prm oznacza współczynnik regularacji L1. Jeżeli wartość tego argumentu to None, to regularyzacja L2 nie zostanie zastosowana. Domyślna wartość arguemntu to None.<br/>
- Argument L2_regularization_prm oznacza współczynnik regularacji L2. Jeżeli wartość tego argumentu to None, to regularyzacja L2 nie zostanie zastosowana. Domyślna wartość arguemntu to None.

Zapomnijmy na razie regularyzacji, zobaczmy jak sieć wytrenuje się przez 10 epok. Wyjątkowo co epokę będziemy testować sieć na zbiorze treningowym, na tym samym, na którym trenujemy. <br/>
![screen3](img/screen3.png)<br/>
Jak widać po 10 epokach sieć osiągneła skuteczność 0.66.

Pomimo braku regularyzacji, sieć osiąga na zbiorze testowym prawi taką samą skutecznosć. Pomimo braku regularyzacji sieć się nie przetrenowała. Nic dziwnego, architektura sieci jest bardzo prosta. Sieć posiada tylko jedną warstwe ukrytą, posiadającą 30 neuronów.</br
![screen5](img/screen5.png)<br/>

---
Stwórzmy teraz nową sieć, o tej samej architekturze, i spróbujmy ja wytrenować z zastosowaniem regularyzacji L2 oraz dropdown'u. Tym razem co epokę będziemy sieć testować na danych testowych, nie używanych podczas trenowania.
~~~ python
my_net2 = Network((784, 30, 10))
my_net2.test_net(train_data), my_net2.test_net(test_data)
~~~
![screen4](img/screen4.png)<br/>
Po 10 epokach sieć osiągneła skuteczność 37.71% na zbiorze testowym.
---
Aby zapisać sieć z pliku, a następnie załadować sieć z pliku, należy użyć funkcji save oraz load.
~~~ python
my_net1.save(add_shape = False, name='my_net1')
loaded_network = Network.load('my_net1.network')
~~~


