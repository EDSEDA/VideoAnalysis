import {registerables, Chart as ChartJS} from 'chart.js';
import {Chart} from "react-chartjs-2";
import type {ChartProps} from "react-chartjs-2";

ChartJS.register(...registerables);

export type LinePlotProps = Pick<ChartProps<'line', (number | [number, number] | null)[], unknown>, 'data'>;

export const SimpleLinePlot = (props: LinePlotProps) => (
    <Chart
        id={props.data.datasets[0].label}
        type='line'
        data={props.data}
        options={{
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: false,
                },
                y: {
                    display: false,
                },
            },
            plugins: {
                legend: {
                    display: false,
                },
            },
        }}
    />
);
