import {registerables, Chart as ChartJS} from 'chart.js';
import {Chart} from "react-chartjs-2";
import type {ChartProps} from "react-chartjs-2";

ChartJS.register(...registerables);

export type BarPlotProps = Pick<ChartProps<'bar', (number | [number, number] | null)[], unknown>, 'data'>;

export const BarPlot = (props: BarPlotProps) => (
    <Chart
        id='barPlot'
        type='bar'
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
