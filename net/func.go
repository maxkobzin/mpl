package net

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/svarlamov/goyhfin"
)

//---------- Структуры нейронной сети ----------

// Структура котировка
type Quote struct {
	Date   time.Time // Дата и время закрытия
	Open   float64   // Цена Open
	High   float64   // Цена High
	Low    float64   // Цена Low
	Close  float64   // Цена Close
	Volume float64   // Объем
}

// Структура пример
type Example struct {
	I []float64 // Входы
	O []float64 // Выходы
	Y []float64 // Выходы, рассчитанные сетью
}

// Структура выборка
type Sample struct {
	E []Example // Выборка примеров
}

// Структура нейрон
type Neuron struct {
	V   float64   // Значение нейрона
	B   float64   // Смещение нейрона
	dB  float64   // Дельта смещения нейрона
	sdB float64   // Сумма дельты смещения нейрона
	S   float64   // Ошибка нейрона
	Avg float64   // Среднее значение нейрона
	Std float64   // Дисперсия значений нейрона
	W   []float64 // Веса нейрона
	dW  []float64 // Дельта весов нейрона
	sdW []float64 // Сумма дельты весов нейрона
}

// Структура слой
type Layer struct {
	N []Neuron // Слой нейронов
}

// Структура нейронная сеть
type Net struct {
	L    []Layer   // Cтруктура слоев сети
	Eps  float64   // Скорость обучения
	Alf  float64   // Момент
	Itr  int       // Количество иттераций в эпохе
	Age  int       // Количество эпох обучения
	Rat  float64   // Граничное отношение относительного прироста ошибки
	Emse []float64 // Ошибка обучения MSE
	ActF Function  // Интерфейс функции активации
}

//---------- Функции активации ----------

// Интерфейс функции активации
type Function interface {
	Activation(float64) float64 // Функция активации
	Derivative(float64) float64 // Производная фунцкии активации
	Max() float64               // Максимум
	Min() float64               // Минимум
}

// Структура и методы Sigmoid
type Sigmoid struct{}

func (f Sigmoid) Activation(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (f Sigmoid) Derivative(x float64) float64 {
	return x * (1 - x)
}

func (f Sigmoid) Max() float64 {
	return 1
}

func (f Sigmoid) Min() float64 {
	return 0
}

// Структура и методы Tanh
type Tanh struct{}

func (f Tanh) Activation(x float64) float64 {
	return (math.Exp(x) - math.Exp(-x)) / (math.Exp(x) + math.Exp(-x))
}

func (f Tanh) Derivative(x float64) float64 {
	return 1 - math.Pow(x, 2)
}

func (f Tanh) Max() float64 {
	return 1
}

func (f Tanh) Min() float64 {
	return -1
}

// Структура и методы ReLU
type ReLU struct{}

func (f ReLU) Activation(x float64) float64 {
	return math.Max(0, x)
}

func (f ReLU) Derivative(x float64) float64 {
	if x >= 0 {
		x = 1
	}
	return x
}

func (f ReLU) Max() float64 {
	return 100
}

func (f ReLU) Min() float64 {
	return 0
}

// Структура и методы LeakyReLU
type LeakyReLU struct{}

func (f LeakyReLU) Activation(x float64) float64 {
	return math.Max(0.1*x, x)
}

func (f LeakyReLU) Derivative(x float64) float64 {
	if x < 0 {
		x = 0.01
	} else {
		x = 1
	}
	return x
}

func (f LeakyReLU) Max() float64 {
	return 100
}

func (f LeakyReLU) Min() float64 {
	return -10
}

//---------- Формирование входных данных ----------

// Получение котировок с помощью тикера Yahoo Finance
func GetQuotesFromYF(ticker string, period string, tf string) []Quote {
	var quotes []Quote
	resp, err := goyhfin.GetTickerData(ticker, period, tf, false)
	if err != nil {
		// NOTE: For library-specific errors, you can check the err against the errors exposed in goyhfin/errors.go
		fmt.Println("Error fetching Yahoo Finance data:", err)
		panic(err)
	}
	for ind := range resp.Quotes {
		if resp.Quotes[ind].Close == 0 {
			continue
		}
		q := Quote{}
		q.Date = resp.Quotes[ind].ClosesAt
		q.Open = resp.Quotes[ind].Open
		q.High = resp.Quotes[ind].High
		q.Low = resp.Quotes[ind].Low
		q.Close = resp.Quotes[ind].Close
		q.Volume = resp.Quotes[ind].Volume
		quotes = append(quotes, q)
	}
	return quotes
}

// Создание выборки из котировок
func MakeSampleFromQuotes(quotes []Quote, period int, periodForecast int, pips float64, twoOut bool, actf Function) Sample {
	smpl := Sample{[]Example{}}
	for i := 1; i < len(quotes)-period-periodForecast-1; i++ {
		I := make([]float64, period*3)
		out := 2
		if !twoOut {
			out = 1
		}
		O := make([]float64, out)
		Y := make([]float64, out)
		// На входы сети будем подавать изменение цен Close, High, Low с предыдущей ценой Close на протяжении period баров
		for j := 0; j < period; j++ {
			I[j] = quotes[i+j].Close - quotes[i+j-1].Close
			I[j+period] = quotes[i+j].High - quotes[i+j-1].Close
			I[j+period*2] = quotes[i+j].Low - quotes[i+j-1].Close
		}
		// Два выхода сети
		if twoOut {
			O[0] = actf.Min()
			O[1] = actf.Min()
			for j := 0; j < periodForecast; j++ {
				// На выход #1 сети будем подавать классификацию 1 - если цена High в течении periodForecast баров достигла последнию цену Close на pips тиков; иначе - 0
				if quotes[i+period+j].High-quotes[i+period-1].Close >= pips {
					O[0] = actf.Max()
				}
				// На выход #2 сети будем подавать классификацию 1 - если цена Close в течении periodForecast баров достигла последнию цену Low на pips тиков; иначе - 0
				if quotes[i+period-1].Close-quotes[i+period+j].Low >= pips {
					O[1] = actf.Max()
				}
			}
			// Один выход сети
		} else {
			avg := (actf.Max() + actf.Min()) / 2
			O[0] = avg
			for j := 0; j < periodForecast; j++ {
				// На выход #1 сети будем подавать классификацию 1 - если цена High в течении periodForecast баров достигла последнию цену Close на pips тиков; иначе - 0
				if quotes[i+period+j].High-quotes[i+period-1].Close >= pips && quotes[i+period-1].Close-quotes[i+period+j].Low >= pips {
					O[0] = avg
					break
				} else if quotes[i+period+j].High-quotes[i+period-1].Close >= pips && O[0] == avg {
					O[0] = actf.Max()
				} else if quotes[i+period+j].High-quotes[i+period-1].Close >= pips && O[0] == actf.Min() {
					O[0] = avg
					break
				}
				// На выход #1 сети будем подавать классификацию 1 - если цена Close в течении periodForecast баров достигла последнию цену Low на pips тиков; иначе - 0
				if quotes[i+period-1].Close-quotes[i+period+j].Low >= pips && O[0] == avg {
					O[0] = actf.Min()
				} else if quotes[i+period-1].Close-quotes[i+period+j].Low >= pips && O[0] == actf.Max() {
					O[0] = avg
					break
				}
			}
		}
		smpl.E = append(smpl.E, Example{I, O, Y})
	}
	return smpl
}

//---------- Методы выборки ----------

// Длина входов выборки
func (smpl Sample) LenInp() int {
	if len(smpl.E) > 0 {
		return len(smpl.E[0].I)
	}
	return 0
}

// Длина выходов выборки
func (smpl Sample) LenOut() int {
	if len(smpl.E) > 0 {
		return len(smpl.E[0].O)
	}
	return 0
}

// Разделение выборки на обучающую и тестовую
func (smpl Sample) TwoSlice(pct float64) (trn Sample, tst Sample) {
	num := int(float64(len(smpl.E)) * pct)
	trn = Sample{smpl.E[:num]}
	tst = Sample{smpl.E[num:]}
	return
}

// Создание сети
func (smpl Sample) MakeNet(hidLayers int, eps float64, alf float64, itr int, age int, rat float64, actf Function) Net {
	if hidLayers < 1 {
		hidLayers = 1
	}
	l := make([]int, 2+hidLayers)
	l[0] = smpl.LenInp()
	l[len(l)-1] = smpl.LenOut()
	x := float64(l[0]) * 1.618
	for i := 1; i < len(l)-1; i++ {
		if int(x) > smpl.LenOut() {
			l[i] = int(x)
			x /= 1.618
		} else {
			l[i] = smpl.LenOut()
		}
	}
	return Net{MakeLayers(l...), eps, alf, itr, age, rat, make([]float64, age+1), actf}
}

//---------- Инициализация нейронной сети и нормализация выборки ----------

// Создание слоев
func MakeLayers(args ...int) []Layer {
	aLayers := make([]Layer, len(args))
	for i, n := range args {
		if i == 0 {
			aLayers[i] = Layer{make([]Neuron, n)}
		} else {
			aLayers[i] = Layer{MakeNeurons(n, len(aLayers[i-1].N))}
		}
	}
	return aLayers
}

// Создание нейронов
func MakeNeurons(len int, lenW int) []Neuron {
	aNeurons := make([]Neuron, len)
	for i := 0; i < len; i++ {
		aNeurons[i] = Neuron{0, 0, 0, 0, 0, 0, 0, make([]float64, lenW), make([]float64, lenW), make([]float64, lenW)}
	}
	return aNeurons
}

// Инициализация весов
func (net *Net) InitW() {
	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)
	for l := 1; l < len(net.L); l++ {
		for i := range net.L[l].N {
			for j := range net.L[l].N[i].W {
				net.L[l].N[i].W[j] = float64(r1.Intn(100))/100.0 - 0.5
			}
			net.L[l].N[i].B = float64(r1.Intn(100))/100.0 - 0.5
		}
	}
}

// Расчет среднего значения и дисперсии выборки
func (net *Net) CalcAvgStdSample(smpl *Sample) {
	for _, e := range smpl.E {
		for i, v := range e.I {
			net.L[0].N[i].Avg += v
		}
	}
	for i := range net.L[0].N {
		net.L[0].N[i].Avg /= float64(len(smpl.E))
	}
	for _, e := range smpl.E {
		for i, v := range e.I {
			net.L[0].N[i].Std += math.Pow(v-net.L[0].N[i].Avg, 2)
		}
	}
	for i := range net.L[0].N {
		net.L[0].N[i].Std = math.Sqrt(net.L[0].N[i].Std / float64(len(smpl.E)))
	}
}

// Нормализация выборки
func (net Net) NormSample(smpl *Sample) {
	for j, e := range smpl.E {
		for i, v := range e.I {
			smpl.E[j].I[i] = (v - net.L[0].N[i].Avg) / math.Max(net.L[0].N[i].Std, 0.00000001)
		}
	}
}

// Копирование среднего значения и дисперсии выборки
func (net *Net) CopyAvgStdSample(source *Net) {
	for i := range net.L[0].N {
		net.L[0].N[i].Avg = source.L[0].N[i].Avg
		net.L[0].N[i].Std = source.L[0].N[i].Std
	}
}

//---------- Стохастическое обучение сети ----------

// Стохастическое обучение сети
func (net *Net) NetStudy(smpl *Sample) {
	net.Emse[0] = net.SampleMSE(smpl)
	for j := 0; j < net.Age; j++ {
		for i := 0; i < net.Itr; i++ {
			net.SampleStudy(smpl)
		}
		net.Emse[j+1] = net.SampleMSE(smpl)
		avg := net.Emse[j]
		for k := 1; k <= 2; k++ {
			avg += net.Emse[int(math.Max(0, float64(j-k)))]
		}
		avg /= 3.0
		if net.Emse[j+1]/avg > net.Rat {
			break
		}
	}
}

// Стохастическое обучение по выборке
func (net *Net) SampleStudy(smpl *Sample) {
	for _, e := range smpl.E {
		net.Forward(e.I)
		net.BackProp(e.O)
		net.UpdateW()
	}
}

// Функция прямого распространения сети
func (net *Net) Forward(inp []float64) {
	for i, v := range inp {
		net.L[0].N[i].V = v
	}
	for l := 1; l < len(net.L); l++ {
		for i, n := range net.L[l].N {
			sum := n.B
			for j, w := range n.W {
				sum += net.L[l-1].N[j].V * w
			}
			net.L[l].N[i].V = net.ActF.Activation(sum)
		}
	}
}

// Функция обратного распространения сети
func (net *Net) BackProp(out []float64) {
	num := len(net.L) - 1
	for i, n := range net.L[num].N {
		net.L[num].N[i].S = (out[i] - n.V) * net.ActF.Derivative(n.V)
	}
	for l := len(net.L) - 2; l > 0; l-- {
		for i, m := range net.L[l].N {
			sum := 0.0
			for _, n := range net.L[l+1].N {
				sum += n.S * n.W[i]
			}
			net.L[l].N[i].S = sum * net.ActF.Derivative(m.V)
		}
	}
}

// Обновление весов
func (net *Net) UpdateW() {
	for l := 1; l < len(net.L); l++ {
		for i, n := range net.L[l].N {
			for j, m := range net.L[l-1].N {
				net.L[l].N[i].dW[j] = net.Eps*n.S*m.V + net.Alf*net.L[l].N[i].dW[j]
				net.L[l].N[i].W[j] += net.L[l].N[i].dW[j]
			}
			net.L[l].N[i].dB = net.Eps*n.S + net.Alf*net.L[l].N[i].dB
			net.L[l].N[i].B += net.L[l].N[i].dB
		}
	}
}

//---------- Оценка обучения сети ----------

// Расчет выходов по выборке
func (net Net) SampleGetResult(smpl *Sample) {
	for j, e := range smpl.E {
		net.Forward(e.I)
		for i, n := range net.L[len(net.L)-1].N {
			smpl.E[j].Y[i] = n.V
		}
	}
}

// Ошибка обучения по выборке
func (net Net) SampleMSE(smpl *Sample) float64 {
	sum := 0.0
	for _, e := range smpl.E {
		net.Forward(e.I)
		for i, n := range net.L[len(net.L)-1].N {
			sum += math.Pow(e.O[i]-n.V, 2)
		}
	}
	return sum / float64(len(net.L[len(net.L)-1].N)*len(smpl.E))
}

// Процент предсказанных классов
func (net Net) SampleErrorCls(smpl *Sample) float64 {
	k := 0
	avg := (net.ActF.Max() + net.ActF.Min()) / 2
	if smpl.LenOut() > 1 {
		for _, e := range smpl.E {
			for i, v := range e.Y {
				if e.O[i] == net.ActF.Max() && v >= avg || e.O[i] == net.ActF.Min() && v < avg {
					k++
				}
			}
		}
	} else {
		third1 := net.ActF.Min() + (net.ActF.Max()-net.ActF.Min())*0.382
		third2 := net.ActF.Max() - (net.ActF.Max()-net.ActF.Min())*0.382
		for _, e := range smpl.E {
			for i, v := range e.Y {
				if e.O[i] == net.ActF.Max() && v >= third2 || e.O[i] == net.ActF.Min() && v <= third1 || e.O[i] == avg && v > third1 && v < third2 {
					k++
				}
			}
		}
	}

	return float64(k) / float64(smpl.LenOut()*len(smpl.E))
}

// Процент предсказанных прогнозов
func (net Net) SampleErrorFor(smpl *Sample) float64 {
	k1, k2 := 0, 0
	avg := (net.ActF.Max() + net.ActF.Min()) / 2
	if smpl.LenOut() > 1 {
		for _, e := range smpl.E {
			for i, v := range e.Y {
				if e.O[i] == net.ActF.Max() {
					k2++
					if v >= avg {
						k1++
					}
				}
			}
		}
	} else {
		third1 := net.ActF.Min() + (net.ActF.Max()-net.ActF.Min())*0.382
		third2 := net.ActF.Max() - (net.ActF.Max()-net.ActF.Min())*0.382
		for _, e := range smpl.E {
			for i, v := range e.Y {
				if e.O[i] == net.ActF.Max() {
					k2++
					if v >= third2 {
						k1++
					}
				} else if e.O[i] == net.ActF.Min() {
					k2++
					if v <= third1 {
						k1++
					}
				}
			}
		}
	}
	return float64(k1) / float64(k2)
}

// Процент верных прогнозов среди предсказанных
func (net Net) SampleErrorUn(smpl *Sample) float64 {
	k1, k2 := 0, 0
	avg := (net.ActF.Max() + net.ActF.Min()) / 2
	if smpl.LenOut() > 1 {
		for _, e := range smpl.E {
			for i, v := range e.Y {
				if v >= avg {
					k2++
					if e.O[i] == net.ActF.Max() {
						k1++
					}
				}
			}
		}
	} else {
		third1 := net.ActF.Min() + (net.ActF.Max()-net.ActF.Min())*0.382
		third2 := net.ActF.Max() - (net.ActF.Max()-net.ActF.Min())*0.382
		for _, e := range smpl.E {
			for i, v := range e.Y {
				if v >= third2 {
					k2++
					if e.O[i] == net.ActF.Max() {
						k1++
					}
				} else if v <= third1 {
					k2++
					if e.O[i] == net.ActF.Min() {
						k1++
					}
				}
			}
		}
	}
	return float64(k1) / float64(k2)
}

// Оценка значимости входного нейрона
func (net Net) EstInp(nmInp []string, chInp []int) {
	l := 1
	est := make([]float64, len(net.L[l].N[0].W))
	for i := 0; i < len(net.L[l].N); i++ {
		for j := 0; j < len(net.L[l].N[i].W); j++ {
			est[j] += math.Pow(net.L[l].N[i].W[j], 2)
		}
	}
	for j := 0; j < len(net.L[l].N[0].W); j++ {
		est[j] = est[j] / float64(len(net.L[l].N))
	}
	for i := 0; i < len(est); i++ {
		fmt.Printf("%s %.4f \n", nmInp[chInp[i]], est[i])
	}
}

//---------- Пакетное обучение сети ----------

// Пакетное обучение сети
func (net Net) NetStudyBatch(smpl *Sample, size int) {
	net.NormBatch(smpl)
	net.Emse[0] = net.SampleBatchMSE(smpl)
	for j := 0; j < net.Age; j++ {
		for i := 0; i < net.Itr; i++ {
			for k := 0; k < len(smpl.E)/size; k++ {
				net.BatchStudy(&Sample{smpl.E[k*size : (k+1)*size]})
			}
			if len(smpl.E)%size > 0 {
				net.BatchStudy(&Sample{smpl.E[(len(smpl.E)/size)*size : len(smpl.E)]})
				net.UpdateWBatch()
			}
		}
		net.NormBatch(smpl)
		net.Emse[j+1] = net.SampleBatchMSE(smpl)
		avg := net.Emse[j]
		for k := 1; k <= 2; k++ {
			avg += net.Emse[int(math.Max(0, float64(j-k)))]
		}
		avg /= 3.0
		if net.Emse[j+1]/avg > net.Rat {
			break
		}
	}
}

// Пакетное обучение по выборке
func (net Net) BatchStudy(smpl *Sample) {
	net.NormBatch(smpl)
	for _, e := range smpl.E {
		net.NormForward(e.I)
		net.BackProp(e.O)
		net.SumdW()
	}
	net.UpdateWBatch()
}

// Нормализация скрытых нейронов
func (net Net) NormBatch(smpl *Sample) {
	for l := 1; l < len(net.L); l++ {
		for i := range net.L[l].N {
			net.L[l].N[i].Avg = 0
			net.L[l].N[i].Std = 1
		}
	}
	for l := 1; l < len(net.L); l++ {
		avg := make([]float64, len(net.L[l].N))
		std := avg
		for _, e := range smpl.E {
			for i := range net.L[l].N {
				net.L[0].N[i].V = e.I[i]
			}
			for k := 1; k < l; k++ {
				for i := 0; i < len(net.L[k].N); i++ {
					net.NeuronNormForward(i, k)
				}
			}
			for i := 0; i < len(net.L[l].N); i++ {
				sum := net.L[l].N[i].B
				for j := 0; j < len(net.L[l].N[i].W); j++ {
					sum += net.L[l-1].N[j].V * net.L[l].N[i].W[j]
				}
				avg[i] += sum
			}
		}
		for i := 0; i < len(net.L[l].N); i++ {
			avg[i] /= float64(len(smpl.E))
		}
		for _, e := range smpl.E {
			for i := range net.L[l].N {
				net.L[0].N[i].V = e.I[i]
			}
			for k := 1; k < l; k++ {
				for i := 0; i < len(net.L[k].N); i++ {
					net.NeuronNormForward(i, k)
				}
			}
			for i := 0; i < len(net.L[l].N); i++ {
				sum := net.L[l].N[i].B
				for j := 0; j < len(net.L[l].N[i].W); j++ {
					sum += net.L[l-1].N[j].V * net.L[l].N[i].W[j]
				}
				std[i] += math.Pow(sum-avg[i], 2)
			}
		}
		for i := 0; i < len(net.L[l].N); i++ {
			net.L[l].N[i].Avg = avg[i]
			net.L[l].N[i].Std = math.Sqrt(std[i] / float64(len(smpl.E)))
		}
	}
}

// Функция прямого распространения сети
func (net Net) NormForward(inp []float64) {
	for i := 0; i < len(net.L[0].N); i++ {
		net.L[0].N[i].V = inp[i]
	}
	for l := 1; l < len(net.L); l++ {
		for i := 0; i < len(net.L[l].N); i++ {
			net.NeuronNormForward(i, l)
		}
	}
}

// Функция прямого распространения нормированного нейрона
func (net Net) NeuronNormForward(i int, l int) {
	sum := net.L[l].N[i].B
	for j := 0; j < len(net.L[l].N[i].W); j++ {
		sum += net.L[l-1].N[j].V * net.L[l].N[i].W[j]
	}
	net.L[l].N[i].V = net.ActF.Activation((sum - net.L[l].N[i].Avg) / math.Max(net.L[l].N[i].Std, 0.00000001))
}

// Суммирование дельты весов
func (net Net) SumdW() {
	for l := 1; l < len(net.L); l++ {
		for i := 0; i < len(net.L[l].N); i++ {
			net.NeuronSumdW(i, l)
		}
	}
}

// Суммирование дельты весов нейрона
func (net Net) NeuronSumdW(i int, l int) {
	for j := 0; j < len(net.L[l].N[i].W); j++ {
		net.L[l].N[i].sdW[j] += net.L[l].N[i].S * net.L[l-1].N[j].V
	}
	net.L[l].N[i].sdB = net.L[l].N[i].S
}

// Пакетное обновление весов
func (net Net) UpdateWBatch() {
	for l := 1; l < len(net.L); l++ {
		for i := 0; i < len(net.L[l].N); i++ {
			net.NeuronUpdateWBatch(i, l)
		}
	}
}

// Пакетное обновление весов нейрона
func (net Net) NeuronUpdateWBatch(i int, l int) {
	for j := 0; j < len(net.L[l].N[i].W); j++ {
		net.L[l].N[i].dW[j] = net.Eps*net.L[l].N[i].sdW[j] + net.Alf*net.L[l].N[i].dW[j]
		net.L[l].N[i].sdW[j] = 0
		net.L[l].N[i].W[j] += net.L[l].N[i].dW[j]
	}
	net.L[l].N[i].dB = net.Eps*net.L[l].N[i].sdB + net.Alf*net.L[l].N[i].dB
	net.L[l].N[i].sdB = 0
	net.L[l].N[i].B += net.L[l].N[i].dB
}

// Расчет выходов по нормализованной выборке
func (net Net) SampleGetResultBatch(smpl *Sample) {
	for j, e := range smpl.E {
		net.NormForward(e.I)
		for i, n := range net.L[len(net.L)-1].N {
			smpl.E[j].Y[i] = n.V
		}
	}
}

// Ошибка обучения по пакетной выборке
func (net Net) SampleBatchMSE(smpl *Sample) float64 {
	sum := 0.0
	for _, e := range smpl.E {
		net.NormForward(e.I)
		for i, n := range net.L[len(net.L)-1].N {
			sum += math.Pow(e.O[i]-n.V, 2)
		}
	}
	return sum / float64(len(net.L[len(net.L)-1].N)*len(smpl.E))
}

//---------- Подготовка, чтение и запись нейронной сети в CSV файл ----------

// Подготовка записи сети в Csv
func (net Net) SaveCsv() [][]string {
	var str [][]string
	s := []string{"L"}
	for l := 0; l < len(net.L); l++ {
		s = append(s, fmt.Sprint(len(net.L[l].N)))
	}
	str = append(str, s)
	for l := 1; l < len(net.L); l++ {
		for i := 0; i < len(net.L[l].N); i++ {
			str = append(str, append([]string{"L" + fmt.Sprint(l) + "_N" + fmt.Sprint(i), fmt.Sprint(net.L[l].N[i].B)}, strings.Split(strings.Trim(fmt.Sprint(net.L[l].N[i].W), "[]"), " ")...))
		}
	}
	str = append(str, []string{"Eps", fmt.Sprint(net.Eps)})
	str = append(str, []string{"Alf", fmt.Sprint(net.Alf)})
	str = append(str, []string{"Itr", fmt.Sprint(net.Itr)})
	str = append(str, []string{"Age", fmt.Sprint(net.Age)})
	str = append(str, []string{"Rat", fmt.Sprint(net.Rat)})
	str = append(str, append([]string{"Emse"}, strings.Split(strings.Trim(fmt.Sprint(net.Emse), "[]"), " ")...))
	return str
}

// Запись CSV файла
func WriteFile(records [][]string, name string) {
	csvFile, err := os.Create(name)
	if err != nil {
		panic(err)
	}
	w := csv.NewWriter(csvFile)
	w.WriteAll(records)
	if err := w.Error(); err != nil {
		fmt.Println("error writing csv:", err)
	}
}

// Чтение CSV файла
func ReadFile(name string) Net {
	file, err := os.Open(name)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	reader.FieldsPerRecord = 1
	reader.Comment = '#'
	reader.Comma = ';'
	record, e := reader.Read()
	if e != nil {
		fmt.Println(e)
	}
	s := strings.Split(record[0], ",")
	fmt.Println(record)
	var l []int
	if s[0] == "L" {
		for j := 1; j < len(s); j++ {
			tmp, _ := strconv.Atoi(s[j])
			l = append(l, tmp)
		}
	}

	net := Net{MakeLayers(l...), 0, 0, 0, 0, 0, []float64{}, Sigmoid{}}
	for l := 1; l < len(net.L); l++ {
		for i := 0; i < len(net.L[l].N); i++ {
			record, e := reader.Read()
			if e != nil {
				fmt.Println(e)
			}
			s = strings.Split(record[0], ",")
			net.L[l].N[i].B, _ = strconv.ParseFloat(s[1], 64)
			for j := 0; j < len(net.L[l].N[i].W); j++ {
				net.L[l].N[i].W[j], _ = strconv.ParseFloat(s[j+2], 64)
			}
		}
	}
	return net
}
