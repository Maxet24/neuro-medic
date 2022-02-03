import config, telebot
from numpy import exp, array, random, dot
import numpy as np
from telebot import types

bot = telebot.TeleBot(config.TOKEN)
currentPosition = {}
symptoms = {}
userTarget = {}
synaptic_weights = np.load('weights.npy')
# 'diagnos1'
# 'train1'

# NEURO
class NeuralNetwork():
    def __init__(self, synaptic_weights):
        self.synaptic_weights = synaptic_weights

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
# NEURO

@bot.message_handler(commands=['start'])
def startMessage(message):
    currentPosition[message.chat.id] = 'menu'
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("💡 Диагностировать")
    item2 = types.KeyboardButton("🤖 Тренировать ИИ")
    item3 = types.KeyboardButton("⚖ Посмотреть веса")
    markup.add(item1, item2, item3)
    bot.send_message(message.chat.id, 'Добро пожаловать в поликлинику. Здесь работают только роботы. Что вам нужно?',
    reply_markup=markup)


@bot.message_handler(content_types=['text'])
def allMessages(message):
    global currentPosition, symptoms, userTarget, synaptic_weights

    # Веса
    if message.text == '⚖ Посмотреть веса':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        item1 = types.KeyboardButton("💡 Диагностировать")
        item2 = types.KeyboardButton("🤖 Тренировать ИИ")
        item3 = types.KeyboardButton("⚖ Посмотреть веса")
        markup.add(item1, item2, item3)
        bot.send_message(message.chat.id, str(synaptic_weights),
    reply_markup=markup)

    if message.text == '💡 Диагностировать':
        synaptic_weights = np.load('weights.npy')
        symptoms[message.chat.id] = []
        userTarget[message.chat.id] = 'diagnos'
        currentPosition[message.chat.id] = 'test1'
        bot.send_message(message.chat.id, 'Какая у вас температура? Укажите число от 35.0 до 42.0 десятичной дробью с точкой.',
         reply_markup=types.ReplyKeyboardRemove())

    elif message.text == '🤖 Тренировать ИИ':
        synaptic_weights = np.load('weights.npy')
        symptoms[message.chat.id] = []
        userTarget[message.chat.id] = 'train'
        currentPosition[message.chat.id] = 'test1'
        bot.send_message(message.chat.id, 'Какая у вас температура? Укажите число от 35.0 до 42.0 десятичной дробью с точкой.',
         reply_markup=types.ReplyKeyboardRemove())
        pass
    
    # 2
    elif currentPosition[message.chat.id] == 'test1':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 36.6')
            return
        if lastInp >= 35.0 and lastInp <= 42.0:
            currentPosition[message.chat.id] = 'test2'
            symptoms[message.chat.id].append((lastInp - 34.0)/8)
            bot.send_message(message.chat.id, 'Сильный ли у вас кашель? Укажите число от 0 до 10.')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #3
    elif currentPosition[message.chat.id] == 'test2':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 10.0:
            currentPosition[message.chat.id] = 'test3'
            symptoms[message.chat.id].append(lastInp/10)
            bot.send_message(message.chat.id, 'Вы чувствуете утомляемость? Укажите число от 0 до 10.')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #4
    elif currentPosition[message.chat.id] == 'test3':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 10.0:
            currentPosition[message.chat.id] = 'test4'
            symptoms[message.chat.id].append(lastInp/10)
            bot.send_message(message.chat.id, 'Есть ли потеря обоняния и вкусовых ощущений? Укажите число от 0 до 1.')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #5
    elif currentPosition[message.chat.id] == 'test4':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 1.0:
            currentPosition[message.chat.id] = 'test5'
            symptoms[message.chat.id].append(lastInp)
            bot.send_message(message.chat.id, 'Есть ли боль в горле? Укажите число от 0 до 10.')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #6
    elif currentPosition[message.chat.id] == 'test5':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 10.0:
            currentPosition[message.chat.id] = 'test6'
            symptoms[message.chat.id].append(lastInp/10)
            bot.send_message(message.chat.id, 'Есть ли головная боль? Укажите число от 0 до 1.')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #7
    elif currentPosition[message.chat.id] == 'test6':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 10.0:
            currentPosition[message.chat.id] = 'test7'
            symptoms[message.chat.id].append(lastInp)
            bot.send_message(message.chat.id, 'Различные другие болевые ощущения? Укажите число от 0 до 10.')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #8
    elif currentPosition[message.chat.id] == 'test7':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 10.0:
            currentPosition[message.chat.id] = 'test8'
            symptoms[message.chat.id].append(lastInp/10)
            bot.send_message(message.chat.id, 'Есть ли диарея? Укажите число от 0 до 1.')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #9
    elif currentPosition[message.chat.id] == 'test8':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 1.0:
            currentPosition[message.chat.id] = 'test9'
            symptoms[message.chat.id].append(lastInp)
            bot.send_message(message.chat.id, 'Есть ли сыпь на коже или изменение цвета кожи на пальцах рук или ног? Укажите число от 0 до 1.')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #10
    elif currentPosition[message.chat.id] == 'test9':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 1.0:
            currentPosition[message.chat.id] = 'test10'
            symptoms[message.chat.id].append(lastInp)
            bot.send_message(message.chat.id, 'Есть ли покраснение или раздражение глаз? Укажите число от 0 до 1.')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #11
    elif currentPosition[message.chat.id] == 'test10':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 1.0:
            currentPosition[message.chat.id] = 'test11'
            symptoms[message.chat.id].append(lastInp)
            bot.send_message(message.chat.id, 'Чувствуете ли затрудненное дыхание или одышку? Укажите число от 0 до 10.')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #12
    elif currentPosition[message.chat.id] == 'test11':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 10.0:
            currentPosition[message.chat.id] = 'test12'
            symptoms[message.chat.id].append(lastInp/10)
            bot.send_message(message.chat.id, 'Чувствуете ли нарушение речи или двигательных функций или спутанность сознания? Укажите число от 0 до 1.')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #13
    elif currentPosition[message.chat.id] == 'test12':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 1.0:
            currentPosition[message.chat.id] = 'test13'
            symptoms[message.chat.id].append(lastInp)
            bot.send_message(message.chat.id, 'Есть ли боль в грудной клетке? Укажите число от 0 до 1.')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    # Реальные показания
    elif currentPosition[message.chat.id] == 'test13' and userTarget[message.chat.id] == 'train':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 1.0:
            currentPosition[message.chat.id] = 'covidTest'
            symptoms[message.chat.id].append(lastInp)
            bot.send_message(message.chat.id, 'Болеете ли вы Короновирусом COVID-19? (отвечать да или нет только после заключения врача)')
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #Конец проверок
    #Диагноз
    elif currentPosition[message.chat.id] == 'test13' and userTarget[message.chat.id] == 'diagnos':
        lastInp = message.text
        try:
            lastInp = float(lastInp)
        except ValueError:
            bot.send_message(message.chat.id, 'Введите число. Пример: 1')
            return
        if lastInp >= 0.0 and lastInp <= 1.0:
            currentPosition[message.chat.id] = 'diagnosEnd'
            symptoms[message.chat.id].append(lastInp)
            neural_network = NeuralNetwork(synaptic_weights)
            print(synaptic_weights)
            print(symptoms[message.chat.id])
            result = neural_network.think(array(symptoms[message.chat.id]))
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            item1 = types.KeyboardButton("💡 Диагностировать")
            item2 = types.KeyboardButton("🤖 Тренировать ИИ")
            item3 = types.KeyboardButton("⚖ Посмотреть веса")
            markup.add(item1, item2, item3)
            bot.send_message(message.chat.id, str(round(result[0] * 100, 1)) + '% - вероятность того, что вы больны COVID-19.',
             reply_markup=markup)
        else:
            bot.send_message(message.chat.id, 'Введите реальные данные.')
            return

    #Тренинг
    elif currentPosition[message.chat.id] == 'covidTest' and userTarget[message.chat.id] == 'train':
        lastInp = message.text.lower()
        if lastInp == 'да' or lastInp == 'нет':
            if lastInp == 'да':
                lastInp = 1.0
            else:
                lastInp = 0.0
            currentPosition[message.chat.id] = 'diagnosEnd'
            print(synaptic_weights)
            print(symptoms[message.chat.id])
            print(lastInp)
            
            neural_network = NeuralNetwork(synaptic_weights)
            training_set_inputs = array([symptoms[message.chat.id]])
            training_set_outputs = array([[lastInp]])
            neural_network.train(training_set_inputs, training_set_outputs, 1)
            print(neural_network.synaptic_weights)

            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            item1 = types.KeyboardButton("💡 Диагностировать")
            item2 = types.KeyboardButton("🤖 Тренировать ИИ")
            item3 = types.KeyboardButton("⚖ Посмотреть веса")
            markup.add(item1, item2, item3)
            bot.send_message(message.chat.id, str(neural_network.synaptic_weights),
             reply_markup=markup)

            np.save('weights', neural_network.synaptic_weights)
        else:
            bot.send_message(message.chat.id, 'Введите да или нет.')
            return

    

        


# Start
bot.polling(non_stop=True)