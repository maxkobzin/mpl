package main

import (
	"fmt"
	"golang-pet/mpl/net"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/olekukonko/tablewriter"
	"github.com/svarlamov/goyhfin"
)

func main() {
	t := time.Now()
	fmt.Println("Старт обработки данных в: ", t)
	// Импортируем котировки
	quotes := net.GetQuotesFromYF("EURUSD=X", goyhfin.OneYear, goyhfin.OneHour)
	writeTableHeadQuotes(quotes, 5)
	// Создадим выборку для нейронной сети на основании  котировок
	actf := net.Sigmoid{}
	gnrSmpl := net.MakeSampleFromQuotes(quotes, 10, 1, 0.002, true, actf)
	writeTableHeadSample(gnrSmpl, 10, actf)
	// Разделим выборку на обучающую 80% и тестовую 20%
	trnSmpl, tstSmpl := gnrSmpl.TwoSlice(0.8)
	// Создадим различные нейронные сети для тестировния
	arrNet := []net.Net{
		gnrSmpl.MakeNet(1, 0.005, 0.05, 100, 10, 0.99, actf),
		gnrSmpl.MakeNet(1, 0.01, 0.1, 100, 10, 0.99, actf),
		gnrSmpl.MakeNet(1, 0.02, 0.2, 100, 10, 0.99, actf),
		gnrSmpl.MakeNet(1, 0.03, 0.3, 100, 10, 0.99, actf),
	}
	// Рассчитаем среднее значение и дисперсию по обучающей выборки
	arrNet[0].CalcAvgStdSample(&trnSmpl)
	// Нормализуем значения обучающей выборки
	arrNet[0].NormSample(&trnSmpl)
	// Нормализуем значения тестовой выборки на основе среднего значения и дисперсии обучающей выборки
	arrNet[0].NormSample(&tstSmpl)
	// Инициализируем начальное значение ошибки err, рассчитаем выходы и ошибку для необученной сети на тестовых данных
	j := 0
	arrNet[j].SampleGetResult(&tstSmpl)
	mse := arrNet[j].SampleMSE(&tstSmpl)
	fmt.Println(", off: ", time.Now())
	fmt.Println("Время обработки выборки: ", time.Since(t))
	t = time.Now()
	fmt.Println("Go runs on: ", t)
	// Обучим нейронные сети
	wg := &sync.WaitGroup{}
	for i := range arrNet {
		wg.Add(1)
		go func(i int, wg *sync.WaitGroup) {
			defer wg.Done()
			// Инициализируем веса
			arrNet[i].InitW()
			// Скопируем ранее расчитанное среднее значение и дисперсию по обучающей выборке
			arrNet[i].CopyAvgStdSample(&arrNet[0])
			// Запускаем стохастическое обучение сети на обучающей выборке
			arrNet[i].NetStudy(&trnSmpl)
			fmt.Printf("Ошибка нейронной сети №%d обучающей выборки: %.4f; тестовой: %.4f\n", i+1, arrNet[i].Emse, arrNet[i].SampleMSE(&tstSmpl))
		}(i, wg)
	}
	wg.Wait()
	// Определим сеть с наименьшей ошибкой
	for i := range arrNet {
		// Рассчитываем ошибку на тестовой выборке и сравниваем со значением mse, для определения сети с наименьшей ошибкой
		err := arrNet[i].SampleMSE(&tstSmpl)
		if err < mse {
			j = i
			mse = err
		}
	}
	// Скопируем сеть с наименьшей ошибкой
	aNet := arrNet[j]
	// Рассчитываем выходы сети после обучения на обучающей выборке
	aNet.SampleGetResult(&trnSmpl)
	// Рассчитываем выходы сети после обучения на тестовой выборке
	aNet.SampleGetResult(&tstSmpl)
	fmt.Println(", off: ", time.Now())
	fmt.Println("Время обучения: ", time.Since(t))
	// Создадим таблицу для вывода результата обучения сети
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Показатель выборки", "Обучающая", "Тестовая"})
	table.SetCaption(true, "Результаты обучения сети")
	table.Append([]string{"Ошибка", strconv.FormatFloat(aNet.SampleMSE(&trnSmpl), 'f', 4, 64), strconv.FormatFloat(aNet.SampleMSE(&tstSmpl), 'f', 4, 64)})
	table.Append([]string{"Процент предсказанных прогнозов", strconv.FormatFloat(aNet.SampleErrorFor(&trnSmpl)*100.0, 'f', 2, 64), strconv.FormatFloat(aNet.SampleErrorUn(&tstSmpl)*100.0, 'f', 2, 64)})
	table.Append([]string{"Процент верных прогнозов среди предсказанных", strconv.FormatFloat(aNet.SampleErrorUn(&trnSmpl)*100.0, 'f', 2, 64), strconv.FormatFloat(aNet.SampleErrorFor(&tstSmpl)*100.0, 'f', 2, 64)})
	table.Append([]string{"Процент верных прогнозов классов", strconv.FormatFloat(aNet.SampleErrorCls(&trnSmpl)*100.0, 'f', 2, 64), strconv.FormatFloat(aNet.SampleErrorCls(&tstSmpl)*100.0, 'f', 2, 64)})
	table.Render()
	// Сохраним сеть для возможности дальнешего использования в файл .csv
	net.WriteFile(aNet.SaveCsv(), "aNET.csv")
	//writeTableHeadSample(trnSmpl, 20, actf)
}

//---------- Вывод табличных данных ----------

// Вывод заголовка и 5 строк таблицы с котировками
func writeTableHeadQuotes(quotes []net.Quote, n int) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Date", "Open", "High", "Low", "Close", "Volume"})
	table.SetCaption(true, "Последние 5 котировок. Всего загружено "+strconv.Itoa(len(quotes))+" котировок.")
	for i, quote := range quotes {
		s := []string{
			quote.Date.Format("2006-01-02 15:04:05"),
			strconv.FormatFloat(quote.Open, 'f', 5, 64),
			strconv.FormatFloat(quote.High, 'f', 5, 64),
			strconv.FormatFloat(quote.Low, 'f', 5, 64),
			strconv.FormatFloat(quote.Close, 'f', 5, 64),
			strconv.FormatFloat(quote.Volume, 'f', 0, 64),
		}
		table.Append(s)
		if i > n-2 {
			break
		}
	}
	table.Render()
}

// Вывод заголовка и 5 строк таблицы с выборкой
func writeTableHeadSample(smpl net.Sample, n int, actf net.Function) {
	table := tablewriter.NewWriter(os.Stdout)
	var head []string
	for i := 0; i < smpl.LenInp() && i < 10; i++ {
		head = append(head, "inp"+strconv.Itoa(i+1))
	}
	for i := 0; i < smpl.LenOut(); i++ {
		head = append(head, "out"+strconv.Itoa(i+1))
	}
	for i := 0; i < smpl.LenOut(); i++ {
		head = append(head, "y"+strconv.Itoa(i+1))
	}
	table.SetHeader(head)
	table.SetCaption(true, "Последние 5 экземляров, первые десять столбцов входов. Всего загружено "+strconv.Itoa(len(smpl.E))+" экземпляров.")
	nm := []string{"Max", "Min", "Avg"}
	msr := make([][3]int, smpl.LenOut())
	for i, example := range smpl.E {
		var ex []string
		for j, inp := range example.I {
			if j < 10 {
				ex = append(ex, strconv.FormatFloat(inp, 'f', 5, 64))
			}
		}
		for _, out := range example.O {
			ex = append(ex, strconv.FormatFloat(out, 'f', 5, 64))
		}
		for _, y := range example.Y {
			ex = append(ex, strconv.FormatFloat(y, 'f', 5, 64))
		}
		table.Append(ex)
		if i > n-2 {
			break
		}
	}
	for _, example := range smpl.E {
		for k, out := range example.O {
			if out == actf.Max() {
				msr[k][0]++
			} else if out == actf.Min() {
				msr[k][1]++
			} else {
				msr[k][2]++
			}
		}
	}
	for j := range nm {
		var footer []string
		for i := 0; i < smpl.LenInp()-1 && i < 9; i++ {
			footer = append(footer, "")
		}
		footer = append(footer, "Total "+nm[j])
		for i := 0; i < smpl.LenOut(); i++ {
			footer = append(footer, strconv.Itoa(msr[i][j]))
		}
		for i := 0; i < smpl.LenOut(); i++ {
			footer = append(footer, "")
		}
		table.Append(footer)
	}
	table.Render()
}
