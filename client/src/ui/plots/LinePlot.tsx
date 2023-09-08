import {registerables, Chart as ChartJS} from 'chart.js';
import {Chart} from "react-chartjs-2";
import type {ChartProps} from "react-chartjs-2";

ChartJS.register(...registerables);

export type LinePlotProps = Pick<ChartProps<'bar', (number | [number, number] | null)[], unknown>, 'data'>;

export const LinePlot = (props: LinePlotProps) => (
    <Chart
        id='linePlot'
        type='line'
        data={props.data}
        options={{
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
            },
        }}
    />
);