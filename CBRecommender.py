# ======================================================================================================================

"""
Our gui implementation will be here
Tkinter will be use
"""

# ======================================================================================================================
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from ast import literal_eval
import cb_recommender_backend as CB
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog     # for combobox, scrolledtext, message box, file dialog, input
from tkinter.ttk import Progressbar

# custom frame sizes
w_column1 = 555
h_column1 = 360

data = {'Can':['The Avengers', 'The Dark Knight'], 'qqqqq':['Memento', 'The Machinist', 'Fight Club']}
recommended = list()
cb = CB.CB()


class DataOptions(tk.Frame):
    """
    Controls the data lists
    """

    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, parent):
        self.parentUserData = parent

        # SET THE FRAME
        frame_File = tk.Frame(master=window, width=w_column1-120, height=h_column1, relief=tk.GROOVE, borderwidth=3)
        frame_File.grid(column=0, row=0, padx=10, pady=10)

        # set title
        label = tk.Label(master=window, text=' Main Menu ', font=('Arial Bold', 10))
        label.place(x=30, y=0)

        # set labels
        label_user = tk.Label(master=window, text="Selected User : ")
        label_user.place(x=30, y=50)
        self.label_selected_user = tk.Label(master=window)
        self.label_selected_user.place(x=120, y=50)

        self.entry_movie = tk.Entry(master=window)
        self.entry_movie.place(x=230, y=50, width=180, height=20)

    # ---------------------------------------------------------------------------------------------------------------------



# ======================================================================================================================

class UserData(tk.Frame):
    def __init__(self, parent):
        self.parentDataOp = parent

        data = {'Can':['The Avengers', 'The Dark Knight'], 'qqqqq':['Memento', 'The Machinist', 'Fight Club']}

        # SET THE UserData FRAME
        frame_UserData = tk.Frame(master=window, width=w_column1-120, height=h_column1, relief=tk.GROOVE, borderwidth=3)
        frame_UserData.grid(column=0, row=1, padx=10, pady=10)

        # set the title
        label = tk.Label(master=window, text=' User Data ', font=('Arial Bold', 10))
        label.place(x=30, y=380)

        # set labels
        label_users = tk.Label(master=window, text="Users")
        label_users.place(x=30, y=410)

        label_movies = tk.Label(master=window, text="Movies")
        label_movies.place(x=230, y=410)

        # set listboxes
        self.listbox_user = tk.Listbox(master=window, width=30, height=13)
        self.listbox_user.place(x=30, y=430)

        self.listbox_user_movies = tk.Listbox(master=window, width=30, height=13)
        self.listbox_user_movies.place(x=230, y=430)

        # set buttons
        button_add_user = tk.Button(master=window, text="Add User", activebackground='gray', command=self.ADD_USER)
        button_add_user.place(x=30, y=642, width=185, height=40)

        button_eject_user = tk.Button(master=window, text="Eject User", activebackground='gray', command=self.EJECT_USER)
        button_eject_user.place(x=30, y=682, width=185, height=40)

        button_add_user_movie = tk.Button(master=window, text="Add Movie", activebackground='gray', command=self.ADD_MOVIE)
        button_add_user_movie.place(x=230, y=642, width=185, height=40)

        button_eject_user_movie = tk.Button(master=window, text="Eject Movie", activebackground='gray', command=self.EJECT_MOVIE)
        button_eject_user_movie.place(x=230, y=682, width=185, height=40)

        self.LOAD_USERS()

        # BINDING
        self.listbox_user.bind("<ButtonRelease-1>", self.BINDING_EVENTS_USER_LIST)





    # ------------------------------------------------------------------------------------------------------------------

    def ADD_USER(self):
        user = simpledialog.askstring('Add User','User: ')
        if user in self.listbox_user.get(0, tk.END):
            messagebox.showerror('Duplicate', 'User already exist')
            return
        self.listbox_user.insert(tk.END, user)
        data[user] = []

    def EJECT_USER(self):
        try:
            removed = self.listbox_user.curselection()[0]
            self.listbox_user.delete(removed)
            data.pop(removed)
        except:
            IndexError()


    def ADD_MOVIE(self):
        try:
            user = self.listbox_user.get(self.listbox_user.curselection()[0])
            movie = simpledialog.askstring('Add Movie', 'Movie: ')
            if movie in data:
                messagebox.showerror('Duplicate', 'Movie alreayd exist')
                return
            data[user].append(movie)
            self.LOAD_USER_MOVIES(user)
        except:
            IndexError()

    def EJECT_MOVIE(self):

        try:
            user = self.parentDataOp.label_selected_user.cget('text')
            movie = self.listbox_user_movies.curselection()[0]
            data[user].pop(movie)
            self.LOAD_USER_MOVIES(user)
        except:
            IndexError()


    def LOAD_USERS(self):
        for user in data:
            self.listbox_user.insert(tk.END, user)

    def LOAD_USER_MOVIES(self, user):
        try:
            self.listbox_user_movies.delete(0, tk.END)
            for movie in data[user]:
                self.listbox_user_movies.insert(tk.END, movie)
        except:
            IndexError()

    # ------------------------------------------------------------------------------------------------------------------
    def BINDING_EVENTS_USER_LIST(self, arg):
        user = self.listbox_user.curselection()[0]
        user = self.listbox_user.get(user)
        self.parentDataOp.label_selected_user.configure(text=user)
        self.LOAD_USER_MOVIES(user)



# ======================================================================================================================

# This class will also handle some multiple events in DataOptions
class Panel(tk.Frame):
    def __init__(self, parentData, parentUser):
        self.parentData = parentData
        self.parentPref = parentUser


        # SET THE Panel FRAME
        frame_panel = tk.Frame(master=window, width=w_column1+40, height=h_column1, relief=tk.GROOVE, borderwidth=3)
        frame_panel.grid(column=1,row=0, padx=10,pady=10)

        label = tk.Label(master=window, text=' Panel ', font=('Arial Bold', 10))
        label.place(x=505, y=0)

        label = tk.Label(master=window, text='Cosine similarities: ')
        label.place(x=500, y=30)

        # set scroll text
        self.scrolltext_cosine_scores = scrolledtext.ScrolledText(master=window, width=65, height=18)
        self.scrolltext_cosine_scores.place(x=500, y=50)

    def UPDATE_PANEL(self):
        self.scrolltext_cosine_scores.delete('1.0', tk.END)
        cosine_scores = cb.keepsimscores
        for movie1, movie2, sim_score in cosine_scores:
            self.scrolltext_cosine_scores.insert(tk.INSERT, "{} - {} ({:.2f})\n".format(movie2, movie1, sim_score))

# ======================================================================================================================

class Model(tk.Frame):
    def __init__(self, parentPanel, parentData, parentUser):
        self.parentPanel = parentPanel
        self.parentData = parentData
        self.parentPref = parentUser

        # set frame
        frame_model = tk.Frame(master=window, width=w_column1+40, height=h_column1, relief=tk.GROOVE, borderwidth=3)
        frame_model.grid(column=1,row=1)

        # set scroll text label
        label = tk.Label(master=window, text=' Recommendations ', font=('Arial Bold', 10))
        label.place(x=505, y=380)

        label = tk.Label(master=window, text='Movies you might like :')
        label.place(x=500, y=415)

        # set scroll text
        self.scrolltext_recommendations = scrolledtext.ScrolledText(master=window, width=65, height=18)
        self.scrolltext_recommendations.place(x=500, y=435)

        # set buttons // DATAOPTIONS
        button_get_recommend_user = tk.Button(master=window, text="Get Recommendations\nBy User",
                                              activebackground='gray', command=self.GET_RECOMMEND_USER)
        button_get_recommend_user.place(x=30, y=80, width=180, height=50)

        button_get_recommend_movie = tk.Button(master=window, text="Get Recommendations\nBy Movie",
                                              activebackground='gray', command=self.GET_RECOMMEND_MOVIE)
        button_get_recommend_movie.place(x=230, y=80, width=180, height=50)

    # ------------------------------------------------------------------------------------------------------------------

    def GET_RECOMMEND_USER(self):
        user = self.parentData.label_selected_user.cget('text')
        movies = data[user]
        recommended = cb.GetRecommend(movies)

        self.scrolltext_recommendations.delete('1.0', tk.END)
        for i in recommended:
            self.scrolltext_recommendations.insert(tk.INSERT, i+'\n')
        self.parentPanel.UPDATE_PANEL()

    def GET_RECOMMEND_MOVIE(self):
        try:
            recommended = cb.GetRecommend([self.parentData.entry_movie.get()])
            self.scrolltext_recommendations.delete('1.0', tk.END)
            for i in recommended:
                self.scrolltext_recommendations.insert(tk.INSERT, i + '\n')
            self.parentPanel.UPDATE_PANEL()
        except:
            KeyError()
            messagebox.showerror('Movie Not Found', "{} not found".format(self.parentData.entry_movie.get()))

# ======================================================================================================================

class Skeleton(tk.Frame):
    def __init__(self, parent):
        self.dataop = DataOptions(window)
        self.userdata = UserData(self.dataop)
        self.panel = Panel(self.dataop, self.userdata)
        self.model = Model(self.panel, self.dataop, self.userdata)


window = tk.Tk()

window.title('Content-Based Recommendation')
window.resizable(False, False)

Skeleton(window)


window.mainloop()
