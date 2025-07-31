import curses
import builtins
import time
import numpy as np

class Simulator:
    def __init__(self):
        # State variables
        self.state = "HOLD"
        self.running = True
        self.prompt_buffer = ""
        self.hold_prompt = ""
        self.input_text = ""

        # ----- Begin curses setup -----
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.nodelay(True)
        h, w = self.stdscr.getmaxyx()
        # status window for logs
        self.status_win = curses.newwin(h - 3, w, 0, 0)
        self.status_win.scrollok(True)
        # input window for user queries
        self.input_win = curses.newwin(3, w, h - 3, 0)
        self.input_win.border()
        self.input_win.nodelay(True)
        # Show initial prompt label
        self.input_win.addstr(1, 2, "Query: ")
        self.input_win.refresh()
        # Monkey-patch print
        self._orig_print = builtins.print
        builtins.print = self.curses_print
        # ----- End curses setup -----

    def curses_print(self, *args, **kwargs):
        msg = ' '.join(str(a) for a in args)
        self.status_win.addstr(msg + "\n")
        self.status_win.refresh()

    def destroy(self):
        # Restore original print and terminal state
        builtins.print = self._orig_print
        curses.nocbreak()
        curses.echo()
        curses.endwin()

    def write_input(self):
        self.input_win.clear()
        self.input_win.border()
        # Always show prompt label
        label = "Query: "
        display = self.input_text
        max_x = self.input_win.getmaxyx()[1] - len(label) - 4
        if len(display) > max_x:
            display = display[-max_x:]
        self.input_win.addstr(1, 2, label + display)
        self.input_win.refresh()

    def run(self):
        last_status_time = 0
        try:
            while self.running:
                now = time.time()
                # Capture key press from input window
                ch = self.input_win.getch()
                key = None
                if ch != -1:
                    if ch in (10, 13):  # Enter
                        key = '\n'
                    elif ch == 27:      # ESC
                        key = '\x1b'
                    elif 32 <= ch <= 126:
                        key = chr(ch)

                # State machine logic
                if self.state == "HOLD":
                    # ESC to LAND
                    if key == '\x1b':
                        print("Esc. Pressed → LANDing...")
                        self.state = "LAND"
                        continue
                    # Enter to submit query
                    if key == '\n':
                        if self.input_text.strip():
                            self.prompt_buffer = self.input_text.strip()
                        self.input_text = ""
                        self.write_input()
                        # SPIN trigger
                        if self.prompt_buffer and self.prompt_buffer != self.hold_prompt:
                            self.hold_prompt = self.prompt_buffer
                            print("Query changed → SPINning to acquire...")
                            self.state = "SPIN"
                        continue
                    # Accumulate text input
                    if key and len(key) == 1:
                        self.input_text += key
                        self.write_input()

                    # Periodic status print every 1s
                    if now - last_status_time > 1:
                        last_status_time = now
                        alt_des = 10.0 + 0.5 * np.sin(now)
                        att_des = 0.2 * np.cos(now)
                        alt_cur = alt_des - 0.3
                        att_cur = att_des + 0.1
                        print(f"Desired Altitude: {alt_des:.2f}, Desired Attitude: {att_des:.2f}")
                        print(f"Current Altitude: {alt_cur:.2f}, Current Attitude: {att_cur:.2f}")

                elif self.state == "SPIN":
                    print("SPINning...")
                    time.sleep(3)
                    print("SPIN complete. Returning to HOLD.")
                    self.state = "HOLD"

                elif self.state == "LAND":
                    for i in range(5, 0, -1):
                        print(f"Landing in {i}...")
                        time.sleep(1)
                    print("LANDING complete. Exiting simulator.")
                    self.running = False

                time.sleep(0.05)

        except KeyboardInterrupt:
            pass
        finally:
            self.destroy()


def main():
    sim = Simulator()
    sim.run()

if __name__ == "__main__":
    main()