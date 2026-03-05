class PID:
    def __init__(
            self, gain_prop: float, gain_int: float, gain_der: float, sensor_period: float,
            output_limits: tuple[float, float]
            ):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        # TODO: define additional attributes you might need
        self.output_limits = output_limits
        self.err_sum = 0.0
        # END OF TODO


    # TODO: implement function which computes the output signal
    # The controller should output only in the range of output_limits
    def output_signal(self, commanded_variable: float, sensor_readings: list[float]) -> float:
        prop = commanded_variable - sensor_readings[0]
        err_now = commanded_variable - sensor_readings[0]
        err_prev = commanded_variable - sensor_readings[1]
        discrete_int = (err_now + err_prev) / 2 * self.sensor_period
        discrete_der = (err_now - err_prev) / self.sensor_period

        self.err_sum += discrete_int
        result = self.gain_prop * prop + \
            self.gain_der * discrete_der + \
            self.gain_int * self.err_sum
        return max(self.output_limits[0], min(result, self.output_limits[1]))
    # END OF TODO
